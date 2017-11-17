#include <cstdio> // Para utilizar archivos: fopen, fread, fwrite, gets...
#include <iostream> // Funciones cout y cin
#include <filesystem>
#include <experimental\filesystem>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2\reg\map.hpp>
#include "opencv2\reg\mappergradshift.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Funciones varias.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;
using namespace experimental::filesystem::v1; //

int main() {

	CalibraRGB();

/// CALIBRACIÓN DE UN SET DE IMÁGENES PARA UNA BANDA MONOCROMÁTICA
/// **********************************************************************************************************************************************************************

	path ruta ("D:/calibracion/");
	string banda = "NIR.TIF";
	int num_img = 0;
	vector<vector<cv::Point2f>> coord_imagen; // Vector de vectores de puntos detectados en las imágenes
	vector<vector<cv::Point3f>> coord_obj; // Vector de vectores de puntos en coordenadas locales (se repiten siempre en cada imagen)
	cv::Mat matrizcam = Mat(Size(3, 3), CV_64F); // Matriz intrínseca de la cámara
	cv::Mat distcoef = Mat(Size(8, 1), CV_64F); // Matriz de coeficientes de distorsión de la cámara
	vector<cv::Mat> rotmat; // Matriz extrínseca de rotaciones de la cámara
	vector<cv::Mat> trasmat; // Matriz extrínseca de traslaciones de la cámara

	for (directory_entry p : recursive_directory_iterator(ruta)) // iteración en carpetas y subcarpetas de la ruta
	{
		path ruta_archivo = p; // Cada archivo o subcarpeta localizado se utiliza como clase path
		string direc = ruta_archivo.string(); // convertir la ruta de clase path a clase string
		string nombre = ruta_archivo.filename().string(); // convertir el nombre del archivo de clase path a string

		if (nombre.size() > 8) // Si el nombre es grande de 8 caracteres significa que es una foto (evita carpetas).
		{
			if (nombre.compare(nombre.size() - 7, 7, banda) == 0) // Encuentra imagen banda buscada
			{
				num_img += 1; // cada vez que entra significa que encuentra una imagen de la banda que quiere

				Mat imagen = cv::imread(direc, CV_LOAD_IMAGE_UNCHANGED); // Para abrir imágenes de 16 bits por píxel CV_LOAD_IMAGE_ANYDEPTH. Para abrir RGB -> CV_LOAD_IMAGE_COLOR
				imagen /= 255; // Para calibrar se necesita convertir a 8 bits por canal
				imagen.convertTo(imagen, CV_8UC1); // Le pongo un canal para el gris.
				cv::Size tamano(12, 8); // número de esquinas a localizar
				vector<cv::Point2f> esquinas; // coordenadas de las esquinas detectadas en la imagen
				
				bool loc = cv::findChessboardCorners(imagen, tamano, esquinas, CV_CALIB_CB_ADAPTIVE_THRESH);
				if (loc == true) // si se localizan bien todas las esquinas es cuando se introduce dentro del set de calibración
				{
					vector<cv::Point3f> obj; // vector de esquinas para la imagen de estudio en coordenadas locales
					for (int j = 1; j <= 8; j++)
					{
						for (int i = 1; i <= 12; i++)
						{
							obj.push_back({ (float(i) - 1.0f) * 30.0f,(float(j) - 1.0f) * 30.0f,0.0f }); // Tablero 30 x 30
						}
					}
					coord_obj.push_back(obj); // introduces las esquinas en coordenadas locales del tablero en el vector general
					coord_imagen.push_back(esquinas); // introduces las esquinas detectadas en coordenadas de la imagen en el vector general


					cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1)); // función para la detección precisa de las esquinas
					cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Se transforma la imagen a 3 canales de color para ponerle la detección de esquinas en color.

					cv::drawChessboardCorners(imagen, tamano, esquinas, loc); // Dibujo de las esquinas detectadas en colores

					cv::imwrite("deteccionesquina//" + std::to_string(num_img) + banda, imagen);
				}
				
			}
		}
		
	}

	double cal = cv::calibrateCamera(coord_obj, coord_imagen, Size(1280,960), matrizcam, distcoef, rotmat, trasmat); // Calibración de la cámara

	double ancho = 4.8; // parámetros obtenidos mediante la multiplicación del tamaño del píxel (3.75 micras para monocromáticas) y de la resolución de la imagen 
	double alto = 3.6; // especifican el ancho y el alto del tamaño del sensor. Probablemente por el tipo de obturador CCD mono y CMOS RGB.
	double fov_x;
	double fov_y;
	double dist_focal;
	cv::Point2d punto_prin;
	double ratio_aspecto;
	
	cv::calibrationMatrixValues(matrizcam, Size(1280, 960), ancho, alto, fov_x, fov_y, dist_focal, punto_prin, ratio_aspecto); // Parámetros físicos de la cámara

	vector<double>matrizfisica; // Vector para obtener la salida de parámetros
	matrizfisica.push_back(dist_focal);
	matrizfisica.push_back(ratio_aspecto);
	matrizfisica.push_back(punto_prin.x);
	matrizfisica.push_back(punto_prin.y);
	matrizfisica.push_back(fov_x);
	matrizfisica.push_back(fov_y);
	matrizfisica.push_back(cal);

	GuardarMat(matrizfisica, "Matriz_Fisica.yml", "Matriz Fisica");

	GuardarMat(matrizcam, "Matriz_Camara.yml", "Matriz Camara");

	GuardarMat(distcoef, "Matriz_Distorsion.yml", "Matriz Distorsion");

	/*
	// Corrección de la distorsión en imágenes de prueba
	cv::Mat prueba = cv::imread("pruebaV.TIF", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
	cv::Mat pruebabien;
	cv::undistort(prueba, pruebabien, matrizcam, distcoef);
	cv::imwrite("pruebaVbien.TIF", pruebabien);
	*/


	// Corrección de la distorsión en imágenes del set de calibrado
	num_img = 0;
	for (directory_entry p : recursive_directory_iterator(ruta))  // iteración en la carpeta
	{
		path ruta_archivo = p; // Cada archivo o subcarpeta localizado se utiliza como clase path
		string direc = ruta_archivo.string(); // convertir la ruta de clase path a clase string
		string nombre = ruta_archivo.filename().string(); // convertir el nombre del archivo de clase path a string

		if (nombre.size() > 8) // Si el nombre es grande de 8 caracteres significa que es una foto (evita carpetas).
		{
			if (nombre.compare(nombre.size() - 7, 7, banda) == 0)
			{
				num_img += 1;
				cv::Mat corregir = cv::imread(direc, CV_LOAD_IMAGE_ANYDEPTH);
				corregir /= 255;
				corregir.convertTo(corregir, CV_8UC1);
				cv::Mat corregida;
				cv::undistort(corregir, corregida, matrizcam, distcoef);
				std::string nombre2 = "set_calibrado_corregido/";
				nombre2.append(to_string(num_img));
				nombre2.append(banda);
				cv::imwrite(nombre2, corregida);
			}
		}
	}
/// **********************************************************************************************************************************************************************
/// **********************************************************************************************************************************************************************

	
	
	
	
	
	
	
	Mat imagenn = cv::imread("nuevacalV7.TIF", CV_LOAD_IMAGE_UNCHANGED);
	imagenn /= 255;
	imagenn.convertTo(imagenn, CV_8UC1);
	cv::Size tamanoo(28, 20); // número de esquinas
	vector<cv::Point3f> objj;
	vector<cv::Point2f> esquinass;
	for (int j = 1; j <= 20; j++)
	{
		for (int i = 1; i <= 28; i++)
		{
			objj.push_back({ (float(i) - 1.0f) * 20.0f,(float(j) - 1.0f) * 20.0f,0.0f });
		}
	}
	bool loc = cv::findChessboardCorners(imagenn, tamanoo, esquinass, CV_CALIB_CB_ADAPTIVE_THRESH);
	cv::cornerSubPix(imagenn, esquinass, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 300, 0.01));

	// Inclusión de 3 canales RGB para el dibujo de las esquinas con color.
	cv::cvtColor(imagenn, imagenn, CV_GRAY2RGB); // Para detectar las esquinas en color

    // Dibujo de las esquinas detectadas en colores
	cv::drawChessboardCorners(imagenn, tamanoo, esquinass, loc);
	
	cv::imwrite("nuevacalVesq7.TIF", imagenn);

	/*
	Mat im1 = imread("pruebaRbienk61MV.TIF", CV_LOAD_IMAGE_UNCHANGED);
	im1 /= 255;
	im1.convertTo(im1, CV_8UC1);
	Mat im2 = imread("pruebaVbienk61.TIF", CV_LOAD_IMAGE_UNCHANGED);
	im2 /= 255;
	im2.convertTo(im2, CV_8UC1);
	Mat mat_transfor(Size(2, 3), CV_64F);
	mat_transfor = estimateRigidTransform(im1, im2, true);
	GuardarMat(mat_transfor, "Cosa.yml", "Cosa");
	//imwrite("alineaRVk61.TIF", resultado);
	*/
	
	/*
	Mat im1 = imread("calibracion/13RojoC.TIF", CV_LOAD_IMAGE_UNCHANGED);
	im1.at<unsigned short>(0, 1189) = 3;
	Mat im2 = imread("calibracion/13VerdeC.TIF", CV_LOAD_IMAGE_UNCHANGED);
	Mat resultado = AlineaImg(im1, im2);
	imwrite("alinea13.TIF", resultado);
	*/

	// Vectores de vectores de puntos. Estructura que utiliza la función de calibración de la cámara
	
	//vector<vector<cv::Point3f>> coord_obj;
	//vector<vector<cv::Point2f>> coord_imagen;
	
	cv::Mat imagen;
	cv::Size tamano(8, 6); // número de esquinas

	for (int i = 1; i <= 18; i++)
	{
		vector<cv::Point3f> obj;
		vector<cv::Point2f> esquinas;

		// Coordenadas del objeto para la imagen 
		for (int j = 1; j <= 6; j++)
		{
			for (int i = 1; i <= 8; i++)
			{
				obj.push_back({ (float(i) - 1.0f) * 28.0f,(float(j) - 1.0f) * 28.0f,0.0f });
			}
		}
		coord_obj.push_back(obj); // introduces en el vector general

		std::string nombre = "calibracion/";
		nombre.append(std::to_string(i));
		nombre.append("Verde.TIF"); // nombre variable

		imagen = cv::imread(nombre, CV_LOAD_IMAGE_UNCHANGED); // Para abrir imágenes de 16 bits por píxel CV_LOAD_IMAGE_ANYDEPTH. Para abrir RGB -> CV_LOAD_IMAGE_COLOR
		// Las imágenes tienen una profundidad de 16 bits por píxel y se cargan con CV_LOAD_IMAGE_UNCHANGED / CV_LOAD_IMAGE_ANYDEPTH. 
		// Es decir 2^16 (0 - 65535) ND, 65536 niveles digitales. Por eso CorelDraw da problemas. depth devuelve 2 al ser CV_16
		// La imagen tiene sólo un canal.
		// Realmente las imágenes tienen 10 bits, pero la almacenarse en paquetes de 1 byte, ocupan 2 bytes = 16 bits.


		// int imgDepth = imagen.depth(); Función que devuelve los bits de la imagen por píxel.
		// imagen.channels(); Función que devuelve los canales de una imagen.
		// Imágenes de 1280 x 960
		// RGB en 4608 x 3456 y en 8 bits por canal y píxel de profundidad

		// Conversión de la imagen a 8 bits
		imagen /= 255;
		imagen.convertTo(imagen, CV_8UC1); // Le pongo un canal para el gris.


		// Localización de las esquinas del tablero.
		bool loc = cv::findChessboardCorners(imagen, tamano, esquinas, CV_CALIB_CB_ADAPTIVE_THRESH);
		cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1));
		// Mejora la detección de las esquinas a escala subpixel. 
		// Parece no tolerar la entrada de valores en double CV_64F, sólo float CV_32F.
		// Si le ponemos que en el algoritmo de iteración que aumente el número mínimo o que epsilon (variación de la solución) se reduzca, la precisión de la localización de las esquinas aumentará
		// Con valores mayores de 30, 0.1 los resultados gráficos dejan de mejorar.

		coord_imagen.push_back(esquinas); // introduces las esquinas en el vector general

		// Inclusión de 3 canales RGB para el dibujo de las esquinas con color.
		cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Para detectar las esquinas en color

		// Dibujo de las esquinas detectadas en colores
		cv::drawChessboardCorners(imagen, tamano, esquinas, loc);

		cv::imwrite("deteccionesquina//" + std::to_string(i) + "Verde.TIF", imagen);

		//cv::namedWindow("hola", CV_WINDOW_FULLSCREEN);
		//cv::imshow("hola", imagen);
		//cv::waitKey(0);
	}
	
	//cv::Mat matrizcam= Mat::eye(3, 3, CV_64F);
	//cv::Mat matrizcam = Mat(Size(3, 3), CV_64F);
	matrizcam.at<double>(0, 0) = 1068.74;
	matrizcam.at<double>(0, 1) = 0.0;
	matrizcam.at<double>(0, 2) = 650.329;
	matrizcam.at<double>(1, 0) = 0.0;
	matrizcam.at<double>(1,1) = 1067.13;
	matrizcam.at<double>(1,2) = 466.17;
	matrizcam.at<double>(2, 0) = 0.0;
	matrizcam.at<double>(2,1) = 0.0;
	matrizcam.at<double>(2,2) = 1.0;

	//matrizcam.at<double>(1, 1) = 1.0; No condiciona nada fijar los valores de partida
	//matrizcam.at<double>(0, 0) = 1.0;

	//cv::Mat distcoef = Mat::zeros(8,1, CV_64F);
	//cv::Mat distcoef = Mat(Size(8, 1), CV_64F);
	distcoef.at<double>(0, 0) = 308.34805117438947;
	distcoef.at<double>(0, 1) = 390.52229314295465;
	distcoef.at<double>(0, 2) = 0.00068590874232127091;
	distcoef.at<double>(0, 3) = -0.00012588044457764470;
	distcoef.at<double>(0, 4) = -315.59251801932652;
	distcoef.at<double>(0, 5) = 308.39330121634094;
	distcoef.at<double>(0, 6) = 507.33006137507221;
	distcoef.at<double>(0, 7) = -290.37421176928223;

	
	//vector<cv::Mat> rotmat;
	//vector<cv::Mat> trasmat;
	//Mat rotmat;
	//Mat trasmat;

	// CV_CALIB_ZERO_TANGENT_DIST Si se quiere anular la distorsión tangencial... es un parámetro pequeño y se supone que el sensor y la lente están bien alineadas en cámaras de calidad.
	// CV_CALIB_FIX_K3 Para fijar k3=0
	// CV_CALIB_FIX_ASPECT_RATIO fx y fy se fijan iguales
	// CV_CALIB_RATIONAL_MODEL para usar componentes de k4, k5 y k6
	// CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5| CV_CALIB_FIX_K6
	// CV_CALIB_USE_INTRINSIC_GUESS
	
	//double cal = cv::calibrateCamera(coord_obj, coord_imagen, imagen.size(), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL);

	
	// cv::Mat matrizcamoptima(cv::Size(3, 3), CV_64F);
	// matrizcamoptima = cv::getOptimalNewCameraMatrix(matrizcam,distcoef,imagen.size(),1); // sirve para conservar todos los píxeles de la imagen original
	// Para utilizarlo se le añade a la función undistort como último parámetro.

	GuardarMat(matrizcam, "Matriz_Camara.yml", "Matriz Camara");

	GuardarMat(distcoef, "Matriz_Distorsion.yml", "Matriz Distorsion");
	
	/*
	for (int i = 1; i <= 18; i++)
	{
		std::string nombre = "calibracion/";
		nombre.append(std::to_string(i));
		nombre.append("NIR.TIF"); // nombre variable
		cv::Mat corregir = cv::imread(nombre, CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
		//cv::imshow("prueba", corregir);
		//cv::waitKey(0);
		cv::Mat corregida;
		cv::undistort(corregir, corregida, matrizcam, distcoef);
		//cv::imshow("prueba", corregida);
		//cv::waitKey(0);
		std::string nombre2 = "calibracion/";
		nombre2.append(std::to_string(i));
		nombre2.append("NIRCMV.TIF"); // nombre variable
		cv::imwrite(nombre2, corregida);
	}
	*/

	for (int i = 1; i <= 18; i++)
	{
		std::string nombre = "calibracion/";
		nombre.append(std::to_string(i));
		nombre.append("Verde.TIF"); // nombre variable
		cv::Mat corregir = cv::imread(nombre, CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
		//cv::imshow("prueba", corregir);
		//cv::waitKey(0);
		cv::Mat corregida;
		cv::undistort(corregir, corregida, matrizcam, distcoef);
		//cv::imshow("prueba", corregida);
		//cv::waitKey(0);
		std::string nombre2 = "calibracion/";
		nombre2.append(std::to_string(i));
		nombre2.append("VerdeC.TIF"); // nombre variable
		cv::imwrite(nombre2, corregida);
	}

	/*
	cv::Mat prueba = cv::imread("pruebaREG.TIF", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
	cv::Mat pruebabien;
	cv::undistort(prueba, pruebabien, matrizcam, distcoef);
	cv::imwrite("pruebaREGbien.TIF", pruebabien);
	*/

	//cv::imshow("prueba", prueba);
	//cv::imshow("pruebabien", pruebabien);
	//cv::waitKey(0);

	
	// Calcular el error de reproyección manualmente.
	// Proyecta los puntos originales con las matrices de transformación de las cámaras (rotación más traslación), hace la diferencia punto a punto y las suma al cuadrado.
	// Sumados todos los puntos de todas las imágenes, divide entre n puntos y hacer la raíz cuadrada.
	// El error de reproyección (medido en píxeles) es un RMSE.
	double error = 0.0;
	int total_puntos = 0;
	for (int i = 0; i <= 17; i++)
	{
		vector<cv::Point2f> coord_obj_proy;
		cv::projectPoints(coord_obj[i], rotmat[i], trasmat[i], matrizcam, distcoef, coord_obj_proy);
		for (int j = 0; j < coord_obj_proy.size(); j++)
		{
			error += (coord_imagen[i][j].x - coord_obj_proy[j].x)*(coord_imagen[i][j].x - coord_obj_proy[j].x) + (coord_imagen[i][j].y - coord_obj_proy[j].y)*(coord_imagen[i][j].y - coord_obj_proy[j].y);
			total_puntos += 1;
		}
	}
	double error_reproy = sqrt(error / total_puntos);
	
	return(0);


}

/*
/// LOCALIZAR MIN Y MAX DE LA MATRIZ
double min, max;
cv::minMaxLoc(imagen, &min, &max);
*/

/*
/// Prueba calibración Agisoft
//cv::Mat matrizcam= Mat::eye(3, 3, CV_64F);
cv::Mat matrizcam2 = Mat(Size(3, 3), CV_64F);
matrizcam2.at<double>(0, 0) = 959.19;
matrizcam2.at<double>(0, 1) = 0.0;
matrizcam2.at<double>(0, 2) = 600.1268;
matrizcam2.at<double>(1, 0) = 0.0;
matrizcam2.at<double>(1, 1) = 959.19;
matrizcam2.at<double>(1, 2) = 415.4909;
matrizcam2.at<double>(2, 0) = 0.0;
matrizcam2.at<double>(2, 1) = 0.0;
matrizcam2.at<double>(2, 2) = 1.0;

//matrizcam.at<double>(1, 1) = 1.0; No condiciona nada fijar los valores de partida
//matrizcam.at<double>(0, 0) = 1.0;

//cv::Mat distcoef = Mat::zeros(8,1, CV_64F);
cv::Mat distcoef2 = Mat(Size(8, 1), CV_64F);
distcoef2.at<double>(0, 0) = -0.177077;
distcoef2.at<double>(0, 1) = 0.047056;
distcoef2.at<double>(0, 2) = 0.00268751;
distcoef2.at<double>(0, 3) = 0.0478248;
distcoef2.at<double>(0, 4) = -0.201056;
distcoef2.at<double>(0, 5) = 0.245166;
distcoef2.at<double>(0, 6) = 0.0;
distcoef2.at<double>(0, 7) = 0.0;

cv::Mat prueba = cv::imread("pruebaV.TIF", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
cv::Mat pruebabien;
cv::undistort(prueba, pruebabien, matrizcam2, distcoef2);
cv::imwrite("pruebaVbien.TIF", pruebabien);
*/