
import json
import logging
import re
from typing import TypedDict, Any
from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_FAST_MODE
from app.knowledge import KnowledgeBase
from app.tools import CommerceTools

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

class AgentState(TypedDict, total=False):
    message: str
    intent: str
    context: str
    tool_result: dict
    answer: str
    sources: list
    escalation: bool

SYSTEM_PROMPT = """You are ShopMate, a careful e-commerce customer support assistant.
Be polite, concise, and transparent. Use only the supplied policy context and tool result.
Never invent order status, refund completion, dates, prices, or policy exceptions.
Never claim a refund or return is completed unless the mock tool explicitly says so.
Ask for missing order ID or missing return reason. Do not request passwords, full card numbers,
CVV, OTPs, or other authentication secrets. If the issue is high-risk, disputed, or unsupported,
offer human escalation. Clearly say this is a demo when relevant."""

logger = logging.getLogger(__name__)

class SupportAgent:
    def __init__(self, data_dir):
        self.kb = KnowledgeBase(data_dir / "faqs.json")
        self.tools = CommerceTools(data_dir)
        self.llm = None
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=300,
                    api_key=GROQ_API_KEY,
                )
            except Exception as exc:
                error_detail = str(exc).replace(GROQ_API_KEY, "[REDACTED]")[:500]
                logger.warning("Groq client initialization failed (%s): %s", type(exc).__name__, error_detail)
                self.llm = None
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def _classify(self, message):
        m = re.sub(r"[\u2010-\u2015\u2212]", "-", message.casefold())
        if re.fullmatch(r"\s*ord[- ]?\d{4,8}\s*", m):
            return "order_status"
        if any(term in m for term in ["password", "passcode", "cvv", "cvc", "otp", "one-time code"]):
            return "privacy"
        if any(term in m for term in ["human", "live agent", "real person", "speak to someone", "complaint", "dispute", "chargeback"]):
            return "escalation"
        if any(term in m for term in ["cancel", "cancellation"]):
            return "cancellation"
        if any(term in m for term in ["return policy", "return window", "eligible for a return", "return exception", "exchange policy"]):
            return "return_policy"
        if any(term in m for term in ["exchange", "swap this", "swap it"]):
            return "exchange"

        scores = {
            "return": sum(term in m for term in [
                "return", "send back", "wrong item", "wrong color", "wrong size",
                "damaged", "defective", "doesn't fit", "does not fit",
            ]),
            "refund": sum(term in m for term in ["refund", "money back", "reimburse", "reimbursement", "bank receive"]),
            "order_status": sum(term in m for term in [
                "where is", "track", "tracking", "order status", "status of", "shipment status",
                "has it shipped", "when will it arrive", "delivery status",
            ]),
            "payment": sum(term in m for term in ["payment", "charged", "billing", "card", "promo", "discount", "duplicate charge"]),
            "shipping": sum(term in m for term in ["shipping", "delivery", "arrive", "ship", "delivered"]),
        }
        best_intent, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score == 0 and re.search(r"\bord[- ]?\d{4,8}\b", m):
            return "order_status"
        return best_intent if best_score else "faq"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("respond", self._respond_node)
        graph.set_entry_point("route")
        graph.add_conditional_edges("route", lambda s: "retrieve",
                       {"respond":"respond", "retrieve":"retrieve"})
        graph.add_edge("retrieve", "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def _route_node(self, state):
        return {"intent": self._classify(state["message"])}
    def _retrieve_node(self, state):
        retrieval_query = state["message"]
        if self._classify(state["message"]) in {"return_policy", "exchange"}:
            retrieval_query = "return exchange window eligibility conditions policy"
        hits = self.kb.search(retrieval_query, 3)
        good = [h for h in hits if h["score"] > 0.05]
        context = "\n".join(f"- {h['title']}: {h['content']}" for h in good)
        sources = [{"title": h["title"], "score": round(h["score"], 3)} for h in good]
        order_match = re.search(r"\bORD[-\u2010-\u2015\u2212 ]?\d{4,8}\b", state["message"].upper())
        if order_match:
            order_id = re.sub(r"[-\u2010-\u2015\u2212 ]", "-", order_match.group(0))
            order_result = self.tools.get_order_status(order_id)
            if order_result["ok"]:
                order = order_result["order"]
                context += (
                    f"\n- Demo order record: order_id={order_id}; status={order.get('status')}; "
                    f"estimated_delivery={order.get('estimated_delivery')}; items={order.get('items')}; "
                    f"return_eligible={order.get('return_eligible')}."
                )
                sources.append({"title": f"Demo order {order_id}", "score": 1.0})
        return {"context": context, "sources": sources}
    def _respond_node(self, state):
        result = self._handle(state["message"], state.get("intent", "faq"), state.get("context", ""),
                              state.get("sources", []))
        return result

    def _handle(self, message, intent, context="", sources=None):
        sources = sources or []
        order_id = re.search(r"\bORD[-\u2010-\u2015\u2212 ]?\d{4,8}\b", message.upper())
        order_id = re.sub(r"[-\u2010-\u2015\u2212 ]", "-", order_id.group(0)) if order_id else None
        if order_id and not self.tools.get_order_status(order_id)["ok"]:
            return {"answer":f"I couldn't find order {order_id} in the demo system. Please check the order ID and try again.",
                    "sources":[], "escalation":False}
        if intent == "order_status":
            if not order_id:
                return {"answer":"I can check that for you. What is your order ID (for example, ORD-1001)? Please do not share payment credentials.",
                        "sources": [], "escalation": False}
            r = self.tools.get_order_status(order_id)
            if not r["ok"]: return {"answer":r["message"], "sources":[], "escalation":False}
            o = r["order"]
            return {"answer":f"Order {order_id}: **{o['status']}**. Estimated delivery: {o.get('estimated_delivery','Not available')}. Items: {o.get('items','Not listed')}. This is demo order data.",
                    "sources":[], "escalation":False}
        if intent == "return":
            if not order_id:
                return {"answer":"I can help start a return. Please provide your order ID (for example, ORD-1001) and a brief reason. Do not include sensitive payment details.",
                        "sources":[], "escalation":False}
            reason_match = re.search(r"(?:because|reason is|reason:)\s*(.+)", message, re.I)
            if not reason_match:
                reason_match = re.search(
                    rf"{re.escape(order_id)}\s*(?:[,;:|-]|for)?\s*(.+)$",
                    message,
                    re.I,
                )
            if not reason_match:
                return {"answer":f"I found the order ID {order_id}. What is the reason for the return (for example, damaged item or wrong size)?",
                        "sources":[], "escalation":False}
            reason = reason_match.group(1).strip(" \t,;:-")
            if not reason:
                return {"answer":f"I found the order ID {order_id}. What is the reason for the return (for example, damaged item or wrong size)?",
                        "sources":[], "escalation":False}
            r = self.tools.create_return_request(order_id, reason)
            return {"answer": (f"Return request **{r['return']['return_id']}** is {r['return']['status'].lower()}. {r['return']['message']}" if r["ok"] else r["message"]),
                    "sources":[], "escalation":False}
        if intent == "refund":
            policy = self.tools.get_refund_policy()
            if order_id:
                status = self.tools.get_order_status(order_id)
                if not status["ok"]: return {"answer":status["message"], "sources":[], "escalation":False}
                o = status["order"]
                if o["status"] == "Refunded":
                    return {"answer":f"Order {order_id} is marked **Refunded** in the demo data. {policy['summary']}", "sources":[], "escalation":False}
                return {"answer":f"Order {order_id} is currently **{o['status']}** in the demo system; I cannot confirm a live payment transaction. {policy['summary']}", "sources":[], "escalation":False}
            return {"answer":policy["summary"] + " If you want me to check the demo order record, share your order ID.", "sources":[], "escalation":False}
        if intent == "exchange":
            if not order_id:
                return {"answer":"An exchange may be possible if the item is unused, in its original condition with included accessories, and within 30 calendar days of delivery. Some categories may be excluded. Please provide your order ID so I can check the demo order record.",
                        "sources":sources, "escalation":False}
            order_result = self.tools.get_order_status(order_id)
            if not order_result["ok"]:
                return {"answer":order_result["message"], "sources":[], "escalation":False}
            order = order_result["order"]
            if order.get("return_eligible") is True:
                answer = f"Order {order_id} ({order.get('items', 'item')}) is marked eligible in the demo record. An exchange may be possible if the item is unused, in its original condition with included accessories, and within 30 calendar days of delivery. A human agent would confirm the exchange details."
            else:
                answer = f"Order {order_id} ({order.get('items', 'item')}) is not marked eligible in the demo record. I can escalate this for human review."
            return {"answer":answer, "sources":sources, "escalation":False}
        if intent == "escalation":
            r = self.tools.escalate_to_human(message)
            return {"answer":f"{r['message']} Reference: **{r['ticket_id']}**. No real support ticket was sent.", "sources":[], "escalation":True}
        if not context.strip():
            return {"answer":"I’m not confident I have a reliable answer for that from the available help documents. I can connect you with human support for review. Would you like me to flag it?",
                    "sources":[], "escalation":False}
        if self.llm and not GROQ_FAST_MODE:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                policy_instruction = ""
                if intent in {"return_policy", "exchange"}:
                    order_context = ""
                    order_match = re.search(r"\bORD[-\u2010-\u2015\u2212 ]?\d{4,8}\b", message.upper())
                    if order_match:
                        normalized_order_id = re.sub(r"[-\u2010-\u2015\u2212 ]", "-", order_match.group(0))
                        order_result = self.tools.get_order_status(normalized_order_id)
                        if order_result["ok"]:
                            order = order_result["order"]
                            order_context = (
                                f"\n\nDemo order record for {normalized_order_id}: "
                                f"status={order.get('status')}; return_eligible={order.get('return_eligible')}; "
                                f"items={order.get('items')}."
                            )
                    policy_instruction = (
                        "This is a return or exchange policy question. Use the return-policy context below "
                        "to answer it. The order ID is only for checking demo eligibility; do not say "
                        "that you lack order-specific return-policy details unless the context truly does not answer the policy question. "
                        "When a demo order record is provided, explicitly state whether that order is marked eligible, "
                        "and explain that eligibility is based on the demo order record."
                        + order_context
                    )
                prompt = f"Customer message: {message}\n\nRetrieved policy context:\n{context}\n\n{policy_instruction}\nAnswer using only this context. If it doesn't answer the question, say so and offer escalation."
                response = self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
                answer = response.content if isinstance(response.content, str) else str(response.content)
                return {"answer":answer, "sources":sources, "escalation":False}
            except Exception as exc:
                error_detail = str(exc).replace(GROQ_API_KEY, "[REDACTED]")[:500]
                logger.warning("Groq request failed; using local fallback (%s): %s", type(exc).__name__, error_detail)
        # Grounded extractive fallback makes the app usable without an API key.
        best = context.splitlines()
        answer = "Here is the relevant information from the help center:\n\n" + "\n".join(best[:2])
        if GROQ_FAST_MODE:
            answer += "\n\nFast mode is enabled, so this answer uses the local demo knowledge base."
        elif GROQ_API_KEY:
            answer += "\n\nGroq could not generate a response, so this answer uses the local demo knowledge base. Check the server log for the error type."
        else:
            answer += "\n\nThis answer uses the local demo knowledge base. Configure GROQ_API_KEY to enable Groq-generated responses."
        return {"answer":answer, "sources":sources, "escalation":False}

    def answer(self, message: str, history=None) -> dict:
        if not message.strip():
            return {"answer":"Please enter a question.", "sources":[], "escalation":False, "intent":"unknown"}
        history = history or []
        order_pattern = r"\bORD[-\u2010-\u2015\u2212 ]?\d{4,8}\b"
        classification_message = message
        current_order = re.search(order_pattern, message.upper())
        recent_order = next(
            (match.group(0) for item in reversed(history)
             for match in [re.search(order_pattern, item.upper())] if match),
            None,
        )
        if not current_order and recent_order:
            classification_message = f"{recent_order} {message}"
        elif re.fullmatch(rf"\s*{order_pattern}\s*", message, re.I):
            previous = next((item for item in reversed(history) if item.strip()), "")
            if previous:
                classification_message = previous
        message_for_handling = message
        if not current_order and recent_order:
            message_for_handling = f"{recent_order} {message}"
        if self.graph:
            try:
                result = self.graph.invoke({"message":classification_message})
                if result.get("intent") in {"return", "order_status", "refund", "exchange"}:
                    intent = result["intent"]
                    result = self._handle(message_for_handling, intent, result.get("context", ""), result.get("sources", []))
                    result["intent"] = intent
            except Exception:
                intent = self._classify(classification_message)
                retrieved = self._retrieve_node({"message":message})
                result = self._handle(message_for_handling, intent, retrieved.get("context",""), retrieved.get("sources",[]))
                result["intent"] = intent
        else:
            intent = self._classify(classification_message)
            retrieved = self._retrieve_node({"message":message})
            result = self._handle(message_for_handling, intent, retrieved.get("context",""), retrieved.get("sources",[]))
            result["intent"] = intent
        result.setdefault("intent", self._classify(message))
        result.setdefault("sources", [])
        result.setdefault("escalation", False)
        return result
