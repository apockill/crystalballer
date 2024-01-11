extends ColorRect

@export var gakken: Node3D
@export var camera: Camera3D


func _process(delta):
	var shadermat = material as ShaderMaterial
	shadermat.set_shader_parameter("camera_up", camera.basis.y)
	shadermat.set_shader_parameter("camera_right", camera.basis.x)
