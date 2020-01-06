#version 330
//layout (location = 1) in vec2 TexCoord;

in vec4 point;
in vec2 texture;
uniform mat4 camera;
out vec2 TexCoord0;

void main() {
    gl_Position = camera * point;
	TexCoord0 = texture;
}
