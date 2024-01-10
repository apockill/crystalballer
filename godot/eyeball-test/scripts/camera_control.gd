@tool
extends Camera3D

@export var distance_ratio: float = 1.666

func _process(_delta):
	var sphere_radius = 0.5;
	var distance = sphere_radius * distance_ratio;
	transform.basis = Basis(Quaternion.from_euler(Vector3(deg_to_rad(-90), 0, 0)))
	position = Vector3(0, distance, 0)
	fov = 2 * rad_to_deg(atan(sphere_radius / distance))
	near = distance
