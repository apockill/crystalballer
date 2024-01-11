extends Node3D

@export var rotate_speed: float = 1


func _process(delta):
	rotate(Vector3.UP, rotate_speed * delta)
