#version 330

in vec2 TexCoord0;

out vec4 outColor;

uniform sampler2D gSampler;

void main() {
//    outColor = vec4(0.5, 0.5, 0, 0);	
	outColor = texture(gSampler,TexCoord0);
}
