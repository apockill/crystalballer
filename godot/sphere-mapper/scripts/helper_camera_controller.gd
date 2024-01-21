extends Node3D

@export var camera: Node3D
@export var pan_speed: float = 0.1
@export var zoom_speed: float = 0.1
@export var min_camera_zoom: float = 0.5

var pan: bool = false


func _input(event):
	if event is InputEventMouseButton:
		var mouse_event = event as InputEventMouseButton
		if mouse_event.button_index == MOUSE_BUTTON_MIDDLE:
			pan = event.is_pressed()
		elif mouse_event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera.position.z = max(min_camera_zoom, camera.position.z - zoom_speed)
		elif mouse_event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera.position.z += zoom_speed
	elif event is InputEventMouseMotion && pan:
		var delta = (event as InputEventMouseMotion).relative
		rotation.x -= deg_to_rad(delta.y * pan_speed)
		rotation.y -= deg_to_rad(delta.x * pan_speed)
