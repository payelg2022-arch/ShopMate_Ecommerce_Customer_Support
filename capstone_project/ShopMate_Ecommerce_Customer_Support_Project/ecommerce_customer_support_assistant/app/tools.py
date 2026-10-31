
"""Mock commerce tools backed by synthetic JSON data. No real payment actions occur."""
import json
from pathlib import Path

class CommerceTools:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.orders = json.loads((data_dir / "orders.json").read_text(encoding="utf-8"))
        self.returns = json.loads((data_dir / "returns.json").read_text(encoding="utf-8"))
    def get_order_status(self, order_id: str):
        order = self.orders.get(order_id.strip().upper())
        if not order:
            return {"ok": False, "message": "I couldn't find that order ID in the demo system. Please check it and try again."}
        return {"ok": True, "order": order}
    def create_return_request(self, order_id: str, reason: str):
        order_id = order_id.strip().upper()
        order = self.orders.get(order_id)
        if not order:
            return {"ok": False, "message": "I couldn't find that order ID. Please verify the ID before requesting a return."}
        if order.get("status") in {"Cancelled", "Return requested", "Refunded"}:
            return {"ok": False, "message": f"A return cannot be created for this order because its status is '{order.get('status')}'."}
        if order.get("return_eligible") is not True:
            return {"ok": False, "message": "This order is not eligible for a return in the demo system. I can escalate this for human review."}
        if order_id in self.returns:
            return {"ok": True, "return": self.returns[order_id], "existing": True}
        record = {"return_id": "RET-" + order_id[-4:], "order_id": order_id,
                  "reason": reason[:240], "status": "Requested",
                  "message": "Return request recorded in the mock system. A human agent would confirm pickup details."}
        self.returns[order_id] = record
        (self.data_dir / "returns.json").write_text(json.dumps(self.returns, indent=2), encoding="utf-8")
        order["status"] = "Return requested"
        (self.data_dir / "orders.json").write_text(json.dumps(self.orders, indent=2), encoding="utf-8")
        return {"ok": True, "return": record, "existing": False}
    def get_refund_policy(self):
        return {"summary": "Eligible refunds are initiated after the returned item passes inspection. Demo estimate: 5–7 business days after approval; bank posting times may vary. This is a policy explanation, not a live payment lookup."}
    def escalate_to_human(self, reason: str):
        return {"escalated": True, "ticket_id": "HUM-" + str(abs(hash(reason)) % 900000 + 100000),
                "message": "I've flagged this for human support. This demo does not send a real ticket."}
