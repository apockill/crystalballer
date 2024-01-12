@tool
extends Camera3D

@export var camera_target: Node3D
@export var gakken_radius: float = 0.5

func _process(_delta):
	look_at(camera_target.global_position)
	
	# calculate fov so that it is tangent to the spheres surface
	# https://www.desmos.com/calculator/vuxczxsxqg
	var distance = (camera_target.global_position - global_position).length()
	fov = 2 * rad_to_deg(acos(sqrt(distance**2 - gakken_radius**2) / distance))
