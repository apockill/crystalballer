shader_type spatial;
render_mode unshaded;

uniform sampler2D mainTex;

void fragment() {
	ALBEDO = texture(mainTex, UV).xyz;
}