from omni.isaac.kit import SimulationApp
import omni.kit.app

simulation_app = SimulationApp({
    "headless": False
})

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaaclab", True)
ext_manager.set_extension_enabled_immediate("isaaclab_tasks", True)

TACEX_PATH = "/workspace/tacex/source"

ext_manager.add_path(TACEX_PATH)

ext_manager.set_extension_enabled_immediate("tacex_uipc", True)
print("Isaac Lab chargé")
print("Simulator is running")

while simulation_app.is_running():
    simulation_app.update() 

simulation_app.close()