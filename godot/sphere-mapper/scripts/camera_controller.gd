@tool
extends Camera3D

@export var camera_target: Node3D


func _process(delta):
	look_at(camera_target.global_position)
	var distance = (camera_target.global_position - global_position).length()
	fov = 2 * rad_to_deg(atan(0.5 / distance))
