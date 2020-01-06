#include <iostream>
#include <vector>

#include "Utility.h"
#include "SOIL.h"

using namespace std;

const uint GRASS_INSTANCES = 10000; // Количество травинок
const int groundsquare = 1;
GL::Camera camera;               // Мы предоставляем Вам реализацию камеры. В OpenGL камера - это просто 2 матрицы. Модельно-видовая матрица и матрица проекции. // ###
                                 // Задача этого класса только в том чтобы обработать ввод с клавиатуры и правильно сформировать эти матрицы.
                                 // Вы можете просто пользоваться этим классом для расчёта указанных матриц.

GLuint texture;
GLuint grassPointsCount; // Количество вершин у модели травинки
GLuint grassShader;      // Шейдер, рисующий траву
GLuint grassVAO;         // VAO для травы (что такое VAO почитайте в доках)
GLuint grassVariance;    // Буфер для смещения координат травинок
vector<VM::vec4> grassVarianceData(GRASS_INSTANCES); // Вектор со смещениями для координат травинок
vector<VM::vec4> grassVarianceOldData(GRASS_INSTANCES); // Вектор со смещениями для координат травинок
GLuint tex_2d;
GLuint groundShader; // Шейдер для земли
GLuint groundVAO;    // VAO для земли

GLuint grassTBO = -1;// Это "фиктивная" текстура для чтения из буфера. На самом деле никакой текстуры не создаётся.
GLuint grassTBO2 = -1;// аналогично

// Размеры экрана
uint screenWidth = 800;
uint screenHeight = 600;
float height = 3;
float ux=0.01, uy=0.02, uz=0.01;//скорость ветра
const int dx = 7;
int counter = 0;
// Это для захвата мышки. Вам это не потребуется (это не значит, что нужно удалять эту строку)
bool captureMouse = true;


// Функция, для привязки к шейдеру TBO - буфера через текстуру
void bindTextureBuffer(GLuint program, GLint unit, const GLchar *name, GLuint texture) {

  glActiveTexture(GL_TEXTURE0 + unit);       CHECK_GL_ERRORS; // делаем активным определённый текстурныю юнит/блок
  glBindTexture(GL_TEXTURE_BUFFER, texture); CHECK_GL_ERRORS; // делаем текущей текстуру отвечающую за наш буфер

  GLint location = glGetUniformLocation(program, name);   CHECK_GL_ERRORS;  // привязываем текущую текстуру (которая фикктивная) и теущим текстурным блоком к имени в шейдере
  if (location >= 0)                                                        // на следующей строчке
    glUniform1i(location, unit);                                            // да да вот тут :)
}


// Функция, рисующая землю
void DrawGround() {
    // Используем шейдер для земли
	glUseProgram(groundShader);                                                  CHECK_GL_ERRORS

		// Устанавливаем юниформ для шейдера. В данном случае передадим перспективную матрицу камеры
		// Находим локацию юниформа 'camera' в шейдере
		GLint cameraLocation = glGetUniformLocation(groundShader, "camera");         CHECK_GL_ERRORS
		// Устанавливаем юниформ (загружаем на GPU матрицу проекции?)                                                     // ###
		glUniformMatrix4fv(cameraLocation, 1, GL_TRUE, camera.getMatrix().data().data()); CHECK_GL_ERRORS

		// Подключаем VAO, который содержит буферы, необходимые для отрисовки земли
		glBindVertexArray(groundVAO);                                                CHECK_GL_ERRORS
		glBindTexture(GL_TEXTURE_2D, texture);
		// Рисуем землю: 2 треугольника (6 вершин)
		glDrawArrays(GL_TRIANGLES, 0, 6);                                            CHECK_GL_ERRORS

		// Отсоединяем VAO
		glBindVertexArray(0);                                                        CHECK_GL_ERRORS
		// Отключаем шейдер
//		bindTextureBuffer(groundShader,1, "Texture/ground.jpeg", tex_2d);
	glUseProgram(0);                                                             CHECK_GL_ERRORS
}

// Обновление смещения травинок
int kx = 1;
int kz = 1;
//int ky = 1;
void UpdateGrassVariance() {
    // Генерация случайных смещений
//	ux = rand() / RAND_MAX / 100 , uy = rand() / RAND_MAX / 100, uz = rand() / RAND_MAX / 100;
	float z = (double)(rand()) / RAND_MAX + 0.07;
	if (ux > z)
		kx = -1;
	if (ux < -z)
		kx = 1;
	if (uz > z)
		kz = -1;
	if (uz < -z)
		kz = 1;
	//if (uy > 0.05)
	//	ky = -1;
	//if (uy < -0.05)
	//	ky = 1;
	//ux = ux + k*float(rand() % (2 * dx) - dx + 1) / 10000.0;
	//uy = uy + k*float(rand() % (2 * dx) - dx + 1) / 10000.0;
	//uz = uz + k*float(rand() % (2 * dx) - dx + 1) / 10000.0;

//	ux = ux + kx*float(rand() % (2 * dx) - dx ) / 2000.0;
//	uy = uy + kx*float(rand() % (2 * dx) - dx ) / 2000.0;
//	uz = uz + kz*float(rand() % (2 * dx) - dx ) / 2000.0;

	ux = ux + kx*float(rand() %  dx) / 8500.0;
//	uy = uy + ky*float(rand() % (2 * dx) - dx ) / 2000.0;
//	uz = uz + kz*float(rand() %  dx) / 6500.0;
	uint i;
	for (i = 0; i < GRASS_INSTANCES; ++i) {
		grassVarianceData[i].x = ux;//+ float(rand() % (2 * dx) - dx) / 1000000.0;;// - (ux/ux)*grassVarianceOldData[i].x;
//		grassVarianceData[i].z = uz;//+ float(rand() % (2 * dx) - dx) / 1000000.0;;// - (uz/uz)*grassVarianceOldData[i].z;
//		grassVarianceData[i].y = uy - k*grassVarianceOldData[i].y;

//		grassVarianceData[i].x = 0.01;
//		grassVarianceData[i].z = 0.01;
//		grassVarianceData[i].y = 0.01;

	}
    // Привязываем буфер, содержащий смещения
    glBindBuffer(GL_ARRAY_BUFFER, grassVariance);                                CHECK_GL_ERRORS
    // Загружаем данные в видеопамять
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * GRASS_INSTANCES, grassVarianceData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
    // Отвязываем буфер
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
}



// Рисование травы
void DrawGrass() {
    // Тут то же самое, что и в рисовании земли
	glUseProgram(grassShader);                                                   CHECK_GL_ERRORS
	GLint cameraLocation = glGetUniformLocation(grassShader, "camera");          CHECK_GL_ERRORS
	glUniformMatrix4fv(cameraLocation, 1, GL_TRUE, camera.getMatrix().data().data()); CHECK_GL_ERRORS
	glBindVertexArray(grassVAO);                                                 CHECK_GL_ERRORS
	// Обновляем смещения для травы
	++counter;
//	if (counter % (INT_MAX - 4) == 3) {
		UpdateGrassVariance();
//		counter = 0;
//	}
	// Отрисовка травинок в количестве GRASS_INSTANCES
    bindTextureBuffer(grassShader, 2, "tboSampler",  grassTBO); // привязываем TBO к имени в шейдере
    bindTextureBuffer(grassShader, 3, "tboSampler2", grassTBO2);// привязываем TBO к имени в шейдере

    glDrawArraysInstanced(GL_TRIANGLES, 0, grassPointsCount, GRASS_INSTANCES);   CHECK_GL_ERRORS
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    glUseProgram(0);                                                             CHECK_GL_ERRORS
}

// Эта функция вызывается для обновления экрана
void RenderLayouts() {
    // Включение буфера глубины
    glEnable(GL_DEPTH_TEST);
    // Очистка буфера глубины и цветового буфера
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Рисуем меши
    DrawGrass();
    DrawGround();
    glutSwapBuffers();
}

// Завершение программы
void FinishProgram() {
    glutDestroyWindow(glutGetWindow());
}

// Обработка события нажатия клавиши (специальные клавиши обрабатываются в функции SpecialButtons)
void KeyboardEvents(unsigned char key, int x, int y) {
    if (key == 27) {
        FinishProgram();
	}
	else if (key == 'w' || key == 'ц') {
        camera.goForward();
    } else if (key == 's' || key == 'ы') {
        camera.goBack();
    } else if (key == 'm' || key == 'ь') {
        captureMouse = !captureMouse;
        if (captureMouse) {
            glutWarpPointer(screenWidth / 2, screenHeight / 2);
            glutSetCursor(GLUT_CURSOR_NONE);
        } else {
            glutSetCursor(GLUT_CURSOR_RIGHT_ARROW);
        }
    }
}

// Обработка события нажатия специальных клавиш
void SpecialButtons(int key, int x, int y) {
    if (key == GLUT_KEY_RIGHT) {
        camera.rotateY(0.02);
    } else if (key == GLUT_KEY_LEFT) {
        camera.rotateY(-0.02);
    } else if (key == GLUT_KEY_UP) {
        camera.rotateTop(-0.02);
    } else if (key == GLUT_KEY_DOWN) {
        camera.rotateTop(0.02);
    }
}

void IdleFunc() {
    glutPostRedisplay();
}

// Обработка события движения мыши
void MouseMove(int x, int y) {
    if (captureMouse) {
        int centerX = screenWidth / 2,
            centerY = screenHeight / 2;
        if (x != centerX || y != centerY) {
            camera.rotateY((x - centerX) / 1000.0f);
            camera.rotateTop((y - centerY) / 1000.0f);
            glutWarpPointer(centerX, centerY);
        }
    }
}

// Обработка нажатия кнопки мыши
void MouseClick(int button, int state, int x, int y) {
}

// Событие изменение размера окна
void windowReshapeFunc(GLint newWidth, GLint newHeight) {
    glViewport(0, 0, newWidth, newHeight);
    screenWidth = newWidth;
    screenHeight = newHeight;

    camera.screenRatio = (float)screenWidth / screenHeight;
}

// Инициализация окна
void InitializeGLUT(int argc, char **argv) {
	glewExperimental = true;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitContextVersion(3, 1);
	glutInitWindowPosition(-1, -1);
    glutInitWindowSize(screenWidth, screenHeight);
    glutCreateWindow("Computer Graphics 3");
    glutWarpPointer(400, 300);
    glutSetCursor(GLUT_CURSOR_NONE);

    glutDisplayFunc(RenderLayouts);
    glutKeyboardFunc(KeyboardEvents);
    glutSpecialFunc(SpecialButtons);
    glutIdleFunc(IdleFunc);
    glutPassiveMotionFunc(MouseMove);
    glutMouseFunc(MouseClick);
    glutReshapeFunc(windowReshapeFunc);
}

// Генерация позиций травинок (эту функцию вам придётся переписать)
vector<VM::vec2> GenerateGrassPositions() {
	vector<VM::vec2> grassPositions(GRASS_INSTANCES);
	for (uint i = 0; i < GRASS_INSTANCES; ++i) {
		grassPositions[i] = VM::vec2(float((double)(rand()) / RAND_MAX * float(groundsquare)), float((double)(rand()) / RAND_MAX * float(groundsquare)));
	}
	return grassPositions;
}
vector<VM::vec2> generateTurns() {
	vector<VM::vec2> turns(GRASS_INSTANCES);
	for (int i = 0; i < GRASS_INSTANCES; i++) {
		turns[i] = VM::vec2((double)(rand()) / RAND_MAX * 2 - 1 , (double)(rand()) / RAND_MAX * 2 - 1);
	}
	return turns;
}
vector<float> generateHeights() {
	vector<float> heights(GRASS_INSTANCES);
	for (int i = 0; i < GRASS_INSTANCES; ++i) {
//		heights[i] = (double)(rand()) / RAND_MAX + 1.3;
		heights[i] = (double)(rand()) / RAND_MAX + 2.5;
	}
	return heights;
}
// Здесь вам нужно будет генерировать меш
vector<VM::vec4> GenMesh(uint n) {
	return{
		VM::vec4(0, 0/height, 0, 1),//0
		VM::vec4(1, 0/height, 0, 1),//1
		VM::vec4(0.5, 1/height, 0, 1),//2
		
		VM::vec4(0.5, 1/height, 0, 1),//2
		VM::vec4(1, 0/height, 0, 1),//1
		VM::vec4(3, 1/height, 0, 1),//3

		VM::vec4(0.5, 1/height, 0, 1),//2
		VM::vec4(3, 1/height, 0, 1),//3
		VM::vec4(2.5/2, 2/height, 0, 1),//4

		VM::vec4(2.5/2, 2/height, 0, 1),//4
		VM::vec4(3, 1/height, 0, 1),//3
		VM::vec4(4, 2/height, 0, 1),//5

		VM::vec4(4, 2/height, 0, 1),//5
		VM::vec4(2.5/2, 2/height, 0, 1),//4
		VM::vec4(6, 3/height, 0, 1),//6
	};
}

// Создание травы
void CreateGrass() {
    uint LOD = 1;
    // Создаём меш
    vector<VM::vec4> grassPoints = GenMesh(LOD);
    // Сохраняем количество вершин в меше травы
    grassPointsCount = grassPoints.size();
    // Создаём позиции для травинок
    vector<VM::vec2> grassPositions = GenerateGrassPositions();
    // Инициализация смещений для травинок
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
        grassVarianceData[i] = VM::vec4(0, 0, 0, 0);
//		grassVarianceOldData[i] = VM::vec4(0.01, 0.01, 0.01, 0.01);
		grassVarianceOldData[i] = VM::vec4(0, 0, 0, 0);
	}

    /* Компилируем шейдеры
    Эта функция принимает на вход название шейдера 'shaderName',
    читает файлы shaders/{shaderName}.vert - вершинный шейдер
    и shaders/{shaderName}.frag - фрагментный шейдер,
    компилирует их и линкует.
    */
    grassShader = GL::CompileShaderProgram("grass");

    // Здесь создаём буфер
	vector<VM::vec2> turns = generateTurns();
	vector<float> heights = generateHeights();
	GLuint pointsBuffer;
    // Это генерация одного буфера (в pointsBuffer хранится идентификатор буфера)
	glGenBuffers(1, &pointsBuffer);                                              CHECK_GL_ERRORS
    // Привязываем сгенерированный буфер
    glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);                                 CHECK_GL_ERRORS
    // Заполняем буфер данными из вектора
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * grassPoints.size(), grassPoints.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

	// Создание VAO
	// Генерация VAO
	glGenVertexArrays(1, &grassVAO);                                             CHECK_GL_ERRORS
	// Привязка VAO
	glBindVertexArray(grassVAO);                                                 CHECK_GL_ERRORS
    // Получение локации параметра 'point' в шейдере
    GLuint pointsLocation = glGetAttribLocation(grassShader, "point");           CHECK_GL_ERRORS
    // Подключаем массив атрибутов к данной локации
    glEnableVertexAttribArray(pointsLocation);                                   CHECK_GL_ERRORS
    // Устанавливаем параметры для получения данных из массива (по 4 значение типа float на одну вершину)
    glVertexAttribPointer(pointsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);          CHECK_GL_ERRORS

	GLuint turnBuffer;
	glGenBuffers(1, &turnBuffer);                                            CHECK_GL_ERRORS
	// Здесь мы привязываем новый буфер, так что дальше вся работа будет с ним до следующего вызова glBindBuffer
	glBindBuffer(GL_ARRAY_BUFFER, turnBuffer);                               CHECK_GL_ERRORS
	glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec2) * turns.size(), turns.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
	GLuint turnlocation = glGetAttribLocation(grassShader, "turn");      CHECK_GL_ERRORS
	glEnableVertexAttribArray(turnlocation);                                 CHECK_GL_ERRORS
	glVertexAttribPointer(turnlocation, 2, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
	// // Здесь мы указываем, что нужно брать новое значение из этого буфера для каждого инстанса (для каждой травинки)
	glVertexAttribDivisor(turnlocation, 1);                                  CHECK_GL_ERRORS

	GLuint heightBuffer;
	glGenBuffers(1, &heightBuffer);                                            CHECK_GL_ERRORS
	// Здесь мы привязываем новый буфер, так что дальше вся работа будет с ним до следующего вызова glBindBuffer
	glBindBuffer(GL_ARRAY_BUFFER, heightBuffer);                               CHECK_GL_ERRORS
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * heights.size(), heights.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
	GLuint heightlocation = glGetAttribLocation(grassShader, "height");      CHECK_GL_ERRORS
	glEnableVertexAttribArray(heightlocation);                                 CHECK_GL_ERRORS
	glVertexAttribPointer(heightlocation, 1, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
	// // Здесь мы указываем, что нужно брать новое значение из этого буфера для каждого инстанса (для каждой травинки)
	glVertexAttribDivisor(heightlocation, 1);                                  CHECK_GL_ERRORS


	// Создаём буфер для позиций травинок
    GLuint positionBuffer;
    glGenBuffers(1, &positionBuffer);                                            CHECK_GL_ERRORS
    // Здесь мы привязываем новый буфер, так что дальше вся работа будет с ним до следующего вызова glBindBuffer
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec2) * grassPositions.size(), grassPositions.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    // GLuint positionLocation = glGetAttribLocation(grassShader, "position");      CHECK_GL_ERRORS
    // glEnableVertexAttribArray(positionLocation);                                 CHECK_GL_ERRORS
    // glVertexAttribPointer(positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    // // Здесь мы указываем, что нужно брать новое значение из этого буфера для каждого инстанса (для каждой травинки)
    // glVertexAttribDivisor(positionLocation, 1);                                  CHECK_GL_ERRORS

    // Создаём буфер для смещения травинок
    glGenBuffers(1, &grassVariance);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, grassVariance);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * GRASS_INSTANCES, grassVarianceData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    //GLuint varianceLocation = glGetAttribLocation(grassShader, "variance");      CHECK_GL_ERRORS
    //glEnableVertexAttribArray(varianceLocation);                                  CHECK_GL_ERRORS
    //glVertexAttribPointer(varianceLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    //glVertexAttribDivisor(varianceLocation, 1);                                  CHECK_GL_ERRORS

    // Отвязываем VAO
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    // Отвязываем буфер
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
//		int width, height;
	//unsigned char* image = SOIL_load_image("Texture/ground.jpg", &width, &height, 0, SOIL_LOAD_RGBA);
	//GLuint texture;
	//glGenTextures(1, &texture);
	//glBindTexture(GL_TEXTURE_2D, texture);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	//glTexParameterf(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameterf(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//	glGenerateMipmap(GL_TEXTURE_2D);
	//SOIL_free_image_data(image);
	//glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done, so we won't accidentily mess up our texture.


    // создание иинициализация фикстивной текстуры для чтения из TBO в шейдере
//	int width, height;
//	unsigned char* image = SOIL_load_image("Texture/grass.jpg", &width, &height, 0, SOIL_LOAD_RGB);
	glGenTextures(1, &grassTBO);                                                 CHECK_GL_ERRORS;
    glBindTexture(GL_TEXTURE_BUFFER, grassTBO);                                  CHECK_GL_ERRORS;
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, positionBuffer);                    CHECK_GL_ERRORS;
    glBindTexture(GL_TEXTURE_BUFFER, 0);	                                       CHECK_GL_ERRORS;

    // ещё раз ... 
    glGenTextures(1, &grassTBO2);                                                 CHECK_GL_ERRORS;
    glBindTexture(GL_TEXTURE_BUFFER, grassTBO2);                                  CHECK_GL_ERRORS;
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, grassVariance);                    CHECK_GL_ERRORS;
    glBindTexture(GL_TEXTURE_BUFFER, 0);	                                        CHECK_GL_ERRORS;

}

// Создаём камеру (Если шаблонная камера вам не нравится, то можете переделать, но я бы не стал)
void CreateCamera() {
    camera.angle = 45.0f / 180.0f * M_PI;
    camera.direction = VM::vec3(0, 0.3, -1);
    camera.position = VM::vec3(float(groundsquare)/2.0, 0.2, 0);
    camera.screenRatio = (float)screenWidth / screenHeight;
    camera.up = VM::vec3(0, 1, 0);
    camera.zfar = 50.0f;
    camera.znear = 0.05f;
}

// Создаём замлю
void CreateGround() {
	// Земля состоит из двух треугольников
	//static const GLfloat meshPoints[] = {
	//	0.0f, 0.0f, 0.0f, 1.0f, 						0.0f, 0.0f,
	//	float(groundsquare), 0.0f, 0.0f, 1.0f, 						1.0f, 0.0f,
	//	float(groundsquare), 0.0f, float(groundsquare), 1.0f, 				1.0f, 1.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f,										0.0f, 0.0f,
	//	float(groundsquare), 0.0f, float(groundsquare), 1.0f,				1.0f, 1.0f,
	//	0.0f, 0.0f, float(groundsquare), 1.0f,							0.0f, 0.0f,
	//};
	static const GLfloat meshPoints[] = {
		0.0f, 0.0f, 0.0f, 1.0f, 						0.0f, 0.0f,
		1.0f, 0.0f, 0.0f, 1.0f, 						1.0f, 0.0f,
		1.0f, 0.0f, 1.0f, 1.0f, 						1.0f, 1.0f,
		0.0f, 0.0f, 0.0f, 1.0f,							0.0f, 0.0f,
		1.0f, 0.0f, 1.0f, 1.0f,							1.0f, 1.0f,
		0.0f, 0.0f, 1.0f, 1.0f,							0.0f, 1.0f,
	};
	// Подробнее о том, как это работает читайте в функции CreateGrass

	groundShader = GL::CompileShaderProgram("ground");
	int width, height;
//	unsigned char* image = SOIL_load_image("Texture/ground.png", &width, &height, 0, SOIL_LOAD_RGB);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	texture = SOIL_load_OGL_texture
	(
		"Texture/ground.png",
		SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
	);
	/* check for an error during the load process */
	if (0 == tex_2d)
	{
		printf("SOIL loading error: '%s'\n", SOIL_last_result());
	}
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
		GL_LINEAR_MIPMAP_LINEAR);

	GLuint pointsBuffer;
	glGenBuffers(1, &pointsBuffer);                                              CHECK_GL_ERRORS
	glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);                                 CHECK_GL_ERRORS
	glBufferData(GL_ARRAY_BUFFER, sizeof(meshPoints), meshPoints, GL_STATIC_DRAW); CHECK_GL_ERRORS

	glGenVertexArrays(1, &groundVAO);                                            CHECK_GL_ERRORS
	glBindVertexArray(groundVAO);                                                CHECK_GL_ERRORS

	GLuint index = glGetAttribLocation(groundShader, "point");                   CHECK_GL_ERRORS
	glEnableVertexAttribArray(index);                                            CHECK_GL_ERRORS
	glVertexAttribPointer(index, 4, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), nullptr);                   CHECK_GL_ERRORS
	index = glGetAttribLocation(groundShader, "texture");                   CHECK_GL_ERRORS
	glEnableVertexAttribArray(index);                                            CHECK_GL_ERRORS
	glVertexAttribPointer(index, 2, GL_FLOAT, GL_FALSE, 6* sizeof(GLfloat),(const void*)(4 * sizeof(GLfloat)));
	glBindVertexArray(0);                                                        CHECK_GL_ERRORS
	glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
	
		//int width, height;
	//unsigned char* image = SOIL_load_image("Texture/ground.jpg", &width, &height, 0, SOIL_LOAD_RGBA);
	//GLuint texture;
	//glGenTextures(1, &texture);
	//glBindTexture(GL_TEXTURE_2D, texture);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glGenerateMipmap(GL_TEXTURE_2D);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
	//	GL_LINEAR_MIPMAP_LINEAR);
	//GLuint index = glGetAttribLocation(groundShader, "texture");                   CHECK_GL_ERRORS
	//glEnableVertexAttribArray(index);                                            CHECK_GL_ERRORS

//	SOIL_free_image_data(image);
}

int main(int argc, char **argv)
{
    try {
        cout << "Start" << endl;
        InitializeGLUT(argc, argv);
        cout << "GLUT inited" << endl;
        glewInit();
        cout << "glew inited" << endl;
        CreateCamera();
        cout << "Camera created" << endl;
        CreateGrass();
        cout << "Grass created" << endl;
        CreateGround();
        cout << "Ground created" << endl;
        glutMainLoop();
    } catch (string s) {
        cout << s << endl;
    }
}
