# 12_api/digital_twin_marketplace.py
import json
from datetime import datetime

class DigitalTwinMarketplace:
    def __init__(self):
        # In-memory registry for developer plugins & custom models
        self.plugin_registry = {}

    def publish_plugin(self, developer_id: str, plugin_name: str, category: str, price_usd: float, code_payload: str) -> dict:
        """Allows third-party developers to upload custom aerospace algorithms/plugins."""
        plugin_id = f"plg_{developer_id}_{int(datetime.utcnow().timestamp())}"
        
        plugin_entry = {
            "plugin_id": plugin_id,
            "developer_id": developer_id,
            "plugin_name": plugin_name,
            "category": category,  # e.g., 'atmospheric_drag', 'thruster_controller', 'ssa_filter'
            "price_usd": price_usd,
            "platform_commission_rate": 0.20, # 20% platform commission
            "status": "ACTIVE",
            "uploaded_at": datetime.utcnow().isoformat(),
            "code_payload": code_payload
        }
        
        self.plugin_registry[plugin_id] = plugin_entry
        return {
            "status": "SUCCESS",
            "message": f"Plugin '{plugin_name}' successfully published to Digital Twin Marketplace.",
            "plugin_id": plugin_id
        }

    def list_available_plugins(self) -> list:
        """Returns all active marketplace plugins for enterprise clients."""
        return [
            {
                "plugin_id": p["plugin_id"],
                "name": p["plugin_name"],
                "developer": p["developer_id"],
                "category": p["category"],
                "price_usd": p["price_usd"]
            }
            for p in self.plugin_registry.values()
        ]

    def purchase_plugin(self, plugin_id: str) -> dict:
        """Handles licensing and revenue split calculation."""
        if plugin_id not in self.plugin_registry:
            return {"status": "ERROR", "message": "Plugin not found."}
        
        plugin = self.plugin_registry[plugin_id]
        commission = plugin["price_usd"] * plugin["platform_commission_rate"]
        developer_payout = plugin["price_usd"] - commission

        return {
            "status": "PURCHASE_SUCCESSFUL",
            "plugin_name": plugin["plugin_name"],
            "total_paid": plugin["price_usd"],
            "platform_revenue": round(commission, 2),
            "developer_payout": round(developer_payout, 2)
        }