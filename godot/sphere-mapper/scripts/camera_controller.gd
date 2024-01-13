@tool
extends Camera3D

@export var camera_target: Node3D
@export var gakken_radius: float = 0.5


func _process(_delta):
	# get the offeset from the gakken to the camera
	var camera_offset = camera_target.global_position - global_position

	# check if offset is aligned with up or down vectors
	if camera_offset.x != 0 || camera_offset.z != 0:
		look_at(camera_target.global_position)
	else:
		global_rotation.x = deg_to_rad(90 * sign(camera_offset.y))

	# calculate fov so that it is tangent to the spheres surface
	# https://www.desmos.com/calculator/vuxczxsxqg
	var distance = camera_offset.length()
	fov = 2 * rad_to_deg(acos(sqrt(distance ** 2 - gakken_radius ** 2) / distance))
