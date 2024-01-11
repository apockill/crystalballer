extends ColorRect

@export var gakken: Node3D
@export var camera: Camera3D


func _process(delta):
	var shadermat = material as ShaderMaterial
	shadermat.set_shader_parameter("gakken_dir", gakken.basis.y)
	shadermat.set_shader_parameter("camera_dir", -camera.basis.z)
