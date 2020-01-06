#version 330

in vec4 point;
//in vec2 position;
//in vec4 variance;
in vec2 turn;
in float height;
uniform mat4 camera;
uniform samplerBuffer tboSampler;
uniform samplerBuffer tboSampler2;


void main() {
    
    //vec2 bladePos = position;
    vec2 bladePos = texelFetch(tboSampler, gl_InstanceID).xy;
    vec4 var      = texelFetch(tboSampler2, gl_InstanceID);

    mat4 scaleMatrix = mat4(1.0);
	
//    scaleMatrix[0][0] = 0.001;//толщина	
	scaleMatrix[0][0] = 0.0012;//толщина	
//  scaleMatrix[0][0] = 0.2;//толщина
    scaleMatrix[1][1] = 0.1;
//	scaleMatrix[1][3] = 2; //высота
	scaleMatrix[1][3] = height; //высота

    mat4 positionMatrix = mat4(1.0);
    
//	positionMatrix[0][1] = 0.5; //углы
//	positionMatrix[0][2] = 0.5;

	positionMatrix[0][1] = turn.x; //углы
	positionMatrix[0][2] = turn.y;
	
	positionMatrix[3][0] = bladePos.x;
    positionMatrix[3][2] = bladePos.y;
	
//    gl_Position = camera * (positionMatrix * scaleMatrix * point + var * point.y*point.y);
    gl_Position = camera * (positionMatrix * scaleMatrix * point + var * point.y*sqrt(point.y));
}
