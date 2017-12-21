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


#include "exiv2\exiv2.hpp"


#include "Funciones varias.h"

#define PI 3.14159265358979323846


///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;
using namespace experimental::filesystem::v1;

///////////////////////////////////////////////////////////////////////////////////////////////////

Mat AlineaImg(Mat& img_base, Mat& img_movida)
{

	// Ajustarlas a 8 bits y un canal.
	img_base /= 255;
	img_base.convertTo(img_base, CV_8UC1); // Le pongo un canal para el gris.
	img_movida /= 255;
	img_movida.convertTo(img_movida, CV_8UC1); // Le pongo un canal para el gris.

								 //imshow("Image 1", im1);
								 //imshow("Image 2", im2);
								 //waitKey(0);

								 // Definir el tipo de movimiento
	//const int warp_mode = MOTION_TRANSLATION;
	const int warp_mode = MOTION_HOMOGRAPHY;


	// Establecer la matriz de la deformación 2x3 or 3x3 según el movimiento.
	Mat warp_matrix;

	// Inicializa la matriz de la deformación como identidad
	if (warp_mode == MOTION_HOMOGRAPHY)
		warp_matrix = Mat::eye(3, 3, CV_32F);
	else
		warp_matrix = Mat::eye(2, 3, CV_32F);

	// Especificar el número de iteraciones
	int number_of_iterations = 5000;

	// Especificar el umbral del incremento
	// en el coeficiente de correlación entre dos iteraciones
	double termination_eps = 1e-10;

	// Definir el criterio de terminación
	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);

	// Ejecutar el algoritmo ECC. Los resultados se almacenan en warp_matrix.
	findTransformECC(
		img_base,
		img_movida,
		warp_matrix,
		warp_mode,
		criteria
	);

	cv::FileStorage guardar("Matriz_Traslacion.yml", cv::FileStorage::WRITE);
	guardar << "matriz inicial" << warp_matrix;
	guardar.release();

	// Para almacenar la imagen alineada
	Mat img_alineada;

	if (warp_mode != MOTION_HOMOGRAPHY)
		// Función warpAffine para traslaciones, transformaciones euclideas y afines.
		warpAffine(img_movida, img_alineada, warp_matrix, img_base.size(), INTER_LINEAR + WARP_INVERSE_MAP);
	else
		// Función warpPerspective para homografías
		warpPerspective(img_movida, img_alineada, warp_matrix, img_base.size(), INTER_LINEAR + WARP_INVERSE_MAP);

	// Mostrar el resultado final
	cv::imshow("Image 1", img_base);
	cv::imshow("Image 2", img_movida);
	cv::imshow("Image 2 Alineada", img_alineada);
	cv::waitKey(0);
	
	return img_alineada;
}


void GuardarMat(Mat& matriz, string ruta, string nombre)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	guardar << nombre << matriz;	
	guardar.release();
}

void GuardarMatCamara(Mat& matriz, string ruta)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	guardar << "Distancia focal X" << matriz.at<double>(0, 0);
	guardar << "Distancia focal Y" << matriz.at<double>(1, 1);
	guardar << "Punto principal X" << matriz.at<double>(0, 2);
	guardar << "Punto principal Y" << matriz.at<double>(1, 2);
	guardar.release();
}

void GuardarMatDistorsion(Mat& matriz, string ruta)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	guardar << "k1" << matriz.at<double>(0, 0);
	guardar << "k2" << matriz.at<double>(0, 1);
	guardar << "p1" << matriz.at<double>(0, 2);
	guardar << "p2" << matriz.at<double>(0, 3);
	guardar << "k3" << matriz.at<double>(0, 4);
	if (matriz.cols - 5 > 0)
	{
		guardar << "k4" << matriz.at<double>(0, 5);
	}
	if (matriz.cols - 6 > 0)
	{
		guardar << "k5" << matriz.at<double>(0, 6);
	}
	if (matriz.cols - 7 > 0)
	{
		guardar << "k6" << matriz.at<double>(0, 7);
	}
	guardar.release();
}

void GuardarMatFisica(vector<double> & vector, string ruta)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	guardar << "Distancia focal" << vector[0];
	guardar << "Relacion de aspecto" << vector[1];
	guardar << "Punto principal X" << vector[2];
	guardar << "Punto principal Y" << vector[3];
	guardar << "Field of View X" << vector[4];
	guardar << "Field of View Y" << vector[5];
	guardar << "Error de reproyeccion" << vector[6];
	guardar.release();
}

void GuardarPuntosDetectados(vector<vector<Point2f>> & coord_img, string ruta)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	for (int i = 0; i < coord_img.size(); i++)
	{
		guardar << "Imagen numero " + to_string(i) << coord_img[i];
	}
}




Mat LeerMatCamara(string ruta_ext)
{
	Mat matrizcam = Mat(Size(3, 3), CV_64F); // Matriz intrínseca de la cámara
	FileStorage leer_matcam(ruta_ext, FileStorage::READ);
	leer_matcam["Distancia focal X"] >> matrizcam.at<double>(0, 0);
	leer_matcam["Distancia focal Y"] >> matrizcam.at<double>(1, 1);
	leer_matcam["Punto principal X"] >> matrizcam.at<double>(0, 2);
	leer_matcam["Punto principal Y"] >> matrizcam.at<double>(1, 2);
	matrizcam.at<double>(0, 1) = 0.0;
	matrizcam.at<double>(1, 0) = 0.0;
	matrizcam.at<double>(2, 0) = 0.0;
	matrizcam.at<double>(2, 1) = 0.0;
	matrizcam.at<double>(2, 2) = 1.0;
	return matrizcam;
}

Mat LeerMatDistorsion(string ruta_ext, int max_k)
{
	Mat distcoef = Mat(Size(2 + max_k, 1), CV_64F); // Matriz de coeficientes de distorsión de la cámara

	FileStorage leer_matdist(ruta_ext, FileStorage::READ);
	leer_matdist["k1"] >> distcoef.at<double>(0, 0);
	leer_matdist["k2"] >> distcoef.at<double>(0, 1);
	leer_matdist["p1"] >> distcoef.at<double>(0, 2);
	leer_matdist["p2"] >> distcoef.at<double>(0, 3);

	if (max_k + 2 > 4)
	{
		leer_matdist["k3"] >> distcoef.at<double>(0, 4);

	}
	if (max_k + 2 > 5)
	{
		leer_matdist["k4"] >> distcoef.at<double>(0, 5);

	}
	if (max_k + 2 > 6)
	{
		leer_matdist["k5"] >> distcoef.at<double>(0, 6);

	}
	if (max_k + 2 > 7)
	{
		leer_matdist["k6"] >> distcoef.at<double>(0, 7);

	}
	return distcoef;
}

vector<vector<Point2f>> LeerPuntosDetectados(string ruta, int num_img)
{
	vector<vector<Point2f>> puntos_leidos;

	FileStorage leer_puntos(ruta, FileStorage::READ);

	for (int i = 0; i < num_img; i++)
	{
		vector<Point2f> puntos_leidos2;
		leer_puntos["Imagen numero " + to_string(i)] >> puntos_leidos2;
		puntos_leidos.push_back(puntos_leidos2);
	}
	return puntos_leidos;
}




void CalibraRGB(string ruta_carpeta_entrada, string& banda_extension, int& num_k, string ruta_salida_deteccionesquina)
{
	/// CALIBRACIÓN DE UN SET DE IMÁGENES PARA BANDA RGB
	/// **********************************************************************************************************************************************************************

	path ruta(ruta_carpeta_entrada);
	int num_img = 0;
	vector<vector<cv::Point2f>> coord_img;
	vector<vector<cv::Point3f>> coord_obj; // Vector de vectores de puntos en coordenadas locales (se repiten siempre en cada imagen)
	cv::Mat matrizcam = Mat(Size(3, 3), CV_64F); // Matriz intrínseca de la cámara
	//matrizcam.at<double>(0, 0) = 1.0; // Para mantener el ratio de aspecto. Se dispara la fx.
	//matrizcam.at<double>(1, 1) = 1.0;

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
			if (nombre.compare(nombre.size() - 7, 7, banda_extension) == 0) // Encuentra imagen banda buscada
			{
				num_img += 1; // cada vez que entra significa que encuentra una imagen de la banda que quiere

				Mat imagen = cv::imread(direc, CV_LOAD_IMAGE_COLOR); // Para abrir imágenes de 16 bits por píxel CV_LOAD_IMAGE_ANYDEPTH. Para abrir BGR -> CV_LOAD_IMAGE_COLOR
				
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
					coord_img.push_back(esquinas); // introduces las esquinas detectadas en coordenadas de la imagen en el vector general

					cv::cvtColor(imagen, imagen, CV_BGR2GRAY);
					
					cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1)); // función para la detección precisa de las esquinas
					
					//cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Se transforma la imagen a 3 canales de color para ponerle la detección de esquinas en color.
					//cv::drawChessboardCorners(imagen, tamano, esquinas, loc); // Dibujo de las esquinas detectadas en colores
					

					//cv::imwrite(ruta_salida_deteccionesquina + std::to_string(num_img) + banda_extension, imagen);
				}

			}
		}

	}

	double error_repro;
	if (num_k == 3)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(4608, 3456), matrizcam, distcoef, rotmat, trasmat); // Calibración de la cámara
	}
	else if (num_k == 2)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(4608, 3456), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_FIX_K3); // Calibración de la cámara
	}															  
	else if (num_k == 4)										  
	{															  
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(4608, 3456), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6); // Calibración de la cámara
	}															   
	else if (num_k == 5)										   
	{															   
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(4608, 3456), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K6); // Calibración de la cámara
	}															
	else if (num_k == 6)										
	{															
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(4608, 3456), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL); // Calibración de la cámara
	}


	double ancho = 6.17472; // parámetros obtenidos mediante la multiplicación del tamaño del píxel (3.75 micras para monocromáticas) y de la resolución de la imagen 
	double alto = 4.63104; // especifican el ancho y el alto del tamaño del sensor. Probablemente por el tipo de obturador CCD mono y CMOS RGB.
	double fov_x;
	double fov_y;
	double dist_focal;
	cv::Point2d punto_prin;
	double ratio_aspecto;

	cv::calibrationMatrixValues(matrizcam, Size(4608, 3456), ancho, alto, fov_x, fov_y, dist_focal, punto_prin, ratio_aspecto); // Parámetros físicos de la cámara

	vector<double>matrizfisica; // Vector para obtener la salida de parámetros
	matrizfisica.push_back(dist_focal);
	matrizfisica.push_back(ratio_aspecto);
	matrizfisica.push_back(punto_prin.x);
	matrizfisica.push_back(punto_prin.y);
	matrizfisica.push_back(fov_x);
	matrizfisica.push_back(fov_y);
	matrizfisica.push_back(error_repro);


	string nombre_salida = banda_extension;
	for (int i = 0; i <= 3; i++) // quitarle el .JPG
	{
		nombre_salida.pop_back();
	}

	string nombre_sal_puntos = nombre_salida;
	nombre_sal_puntos.append(".yml");
	string salida = "Listado_Puntos_";
	salida.append(nombre_sal_puntos);
	GuardarPuntosDetectados(coord_img, salida); // salida = Listado_Puntos_RGB.yml

	nombre_salida.append("_k");
	nombre_salida.append(to_string(num_k));
	nombre_salida.append(".yml");

	salida = "Matriz_Fisica_";
	salida.append(nombre_salida);
	GuardarMatFisica(matrizfisica, salida);

	salida = "Matriz_Camara_";
	salida.append(nombre_salida);
	GuardarMatCamara(matrizcam, salida);

	salida = "Matriz_Distorsion_";
	salida.append(nombre_salida);
	GuardarMatDistorsion(distcoef, salida);

	/*
	// Corrección de la distorsión en imágenes de prueba
	cv::Mat prueba = cv::imread("pruebaRGB.JPG", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
	cv::Mat pruebabien;
	cv::undistort(prueba, pruebabien, matrizcam, distcoef);
	cv::imwrite("pruebaRGBbien.JPG", pruebabien);


	// Corrección de la distorsión en imágenes del set de calibrado
	num_img = 0;
	for (directory_entry p : recursive_directory_iterator(ruta))  // iteración en la carpeta
	{
		path ruta_archivo = p; // Cada archivo o subcarpeta localizado se utiliza como clase path
		string direc = ruta_archivo.string(); // convertir la ruta de clase path a clase string
		string nombre = ruta_archivo.filename().string(); // convertir el nombre del archivo de clase path a string

		if (nombre.size() > 8) // Si el nombre es grande de 8 caracteres significa que es una foto (evita carpetas).
		{
			if (nombre.compare(nombre.size() - 7, 7, "RGB.JPG") == 0)
			{
				num_img += 1;
				cv::Mat corregir = cv::imread(direc, CV_LOAD_IMAGE_COLOR);
				cv::Mat corregida;
				cv::undistort(corregir, corregida, matrizcam, distcoef);
				std::string nombre2 = "set_calibrado_corregido/";
				nombre2.append(to_string(num_img));
				nombre2.append("RGB.JPG");
				cv::imwrite(nombre2, corregida);
			}
		}
	}
	*/
	/// **********************************************************************************************************************************************************************
	/// **********************************************************************************************************************************************************************
}

void CalibraMono(string ruta_carpeta_entrada, string& banda_extension, int& num_k, string ruta_salida_deteccionesquina)
{
	/// CALIBRACIÓN DE UN SET DE IMÁGENES PARA UNA BANDA MONOCROMÁTICA
	/// **********************************************************************************************************************************************************************

	path ruta(ruta_carpeta_entrada); //"D:/calibracion/"
	int num_img = 0;
	vector<vector<cv::Point2f>> coord_img;
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
			if (nombre.compare(nombre.size() - 7, 7, banda_extension) == 0) // Encuentra imagen banda buscada
			{
				num_img += 1; // cada vez que entra significa que encuentra una imagen de la banda que quiere

				Mat imagen = cv::imread(direc, CV_LOAD_IMAGE_UNCHANGED); // Para abrir imágenes de 16 bits por píxel CV_LOAD_IMAGE_ANYDEPTH. Para abrir RGB -> CV_LOAD_IMAGE_COLOR
				imagen.convertTo(imagen, CV_8UC1, 0.0038910505836576); // Le pongo un canal para el gris. El factor de escala (1/257) reduce el 65535 máximo valor en profundidad de 16 bits a un ND de 255.
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
					coord_img.push_back(esquinas); // introduces las esquinas detectadas en coordenadas de la imagen en el vector general


					cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1)); // función para la detección precisa de las esquinas
					//cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Se transforma la imagen a 3 canales de color para ponerle la detección de esquinas en color.

					//cv::drawChessboardCorners(imagen, tamano, esquinas, loc); // Dibujo de las esquinas detectadas en colores

					//cv::imwrite(ruta_salida_deteccionesquina + std::to_string(num_img) + banda_extension, imagen);
				}

			}
		}

	}


	/// POSBILES CALIBRACIONES SEGÚN EL NÚMERO DE PARÁMETROS DE DISTORSIÓN RADIAL QUE SE QUIERAN UTILIZAR
	double error_repro;
	if (num_k == 3)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(1280, 960), matrizcam, distcoef, rotmat, trasmat); // Calibración de la cámara
	}
	else if (num_k == 2) // cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(1280, 960), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_FIX_K3); // Calibración de la cámara
	}
	else if (num_k == 4)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(1280, 960), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6); // Calibración de la cámara
	}
	else if (num_k == 5)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(1280, 960), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K6); // Calibración de la cámara
	}
	else if (num_k == 6)
	{
		error_repro = cv::calibrateCamera(coord_obj, coord_img, Size(1280, 960), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL); // Calibración de la cámara
	}

	double ancho = 4.8; // parámetros obtenidos mediante la multiplicación del tamaño del píxel (3.75 micras para monocromáticas) y de la resolución de la imagen 
	double alto = 3.6; // especifican el ancho y el alto del tamaño del sensor. Probablemente por el tipo de obturador CCD mono y CMOS RGB.
	double fov_x;
	double fov_y;
	double dist_focal;
	cv::Point2d punto_prin;
	double ratio_aspecto;

	cv::calibrationMatrixValues(matrizcam, Size(1280, 960), ancho, alto, fov_x, fov_y, dist_focal, punto_prin, ratio_aspecto); // Parámetros físicos de la cámara

	/// OBTENCIÓN DE LAS MATRICES DE CALIBRACIÓN
	vector<double>matrizfisica; // Vector para obtener la salida de parámetros
	matrizfisica.push_back(dist_focal);
	matrizfisica.push_back(ratio_aspecto);
	matrizfisica.push_back(punto_prin.x);
	matrizfisica.push_back(punto_prin.y);
	matrizfisica.push_back(fov_x);
	matrizfisica.push_back(fov_y);
	matrizfisica.push_back(error_repro);

	
	string nombre_salida = banda_extension;
	for (int i = 0; i <= 3; i++) // quitarle el .TIF a las bandas
	{
		nombre_salida.pop_back();
	}

	string nombre_sal_puntos = nombre_salida;
	nombre_sal_puntos.append(".yml");
	string salida = "Listado_Puntos_";
	salida.append(nombre_sal_puntos);
	GuardarPuntosDetectados(coord_img, salida); // salida = Listado_Puntos_GRE.yml


	nombre_salida.append("_k");
	nombre_salida.append(to_string(num_k));
	nombre_salida.append(".yml");
	
	salida = "Matriz_Fisica_";
	salida.append(nombre_salida);
	GuardarMatFisica(matrizfisica, salida); // salida = Matriz_Fisica_GRE_k3.yml

	salida = "Matriz_Camara_";
	salida.append(nombre_salida);
	GuardarMatCamara(matrizcam, salida);

	salida = "Matriz_Distorsion_";
	salida.append(nombre_salida);
	GuardarMatDistorsion(distcoef, salida);

	/*
	// Corrección de la distorsión en imágenes de prueba
	cv::Mat prueba = cv::imread("pruebaV.TIF", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
	cv::Mat pruebabien;
	cv::undistort(prueba, pruebabien, matrizcam, distcoef);
	cv::imwrite("pruebaVbien.TIF", pruebabien);
	*/

	/*
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
	*/

	/// **********************************************************************************************************************************************************************
	/// **********************************************************************************************************************************************************************
}

void CorrigeImagenes(Mat& mat_cam, Mat& dist_coef, string& banda, string ruta_img_entrada, string ruta_img_salida)
{
	path ruta(ruta_img_entrada); // "D:/calibracion/"
	int num_img = 0;

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
				cv::Mat corregir = cv::imread(direc, CV_LOAD_IMAGE_ANYDEPTH); // En el caso de TIF con 10 bits de profundidad
				corregir.convertTo(corregir, CV_8UC1,1/257);
				cv::Mat corregida;
				cv::undistort(corregir, corregida, mat_cam, dist_coef);
				std::string nombre2 = ruta_img_salida;
				string nombre = banda;
				for (int i = 0; i <= 3; i++)
				{
					nombre.pop_back();
				}
				int num_k = dist_coef.size().width-2;
				nombre2.append(to_string(num_img) + nombre + "_k" + to_string(num_k) + ".TIF");

				cv::imwrite(nombre2, corregida);
			}
		}
	}
}

void CorrigeImagenesRGB(Mat& mat_cam, Mat& dist_coef, string& banda, string ruta_img_entrada, string ruta_img_salida)
{
	path ruta(ruta_img_entrada); // "D:/calibracion/"
	int num_img = 0;

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
				cv::Mat corregir = cv::imread(direc, CV_LOAD_IMAGE_COLOR); // En el caso de JPG son 8 bits de profundidad en RGB.
				cv::Mat corregida;
				cv::undistort(corregir, corregida, mat_cam, dist_coef);
				std::string nombre2 = ruta_img_salida;
				string nombre = banda;
				for (int i = 0; i <= 3; i++)
				{
					nombre.pop_back();
				}
				int num_k = dist_coef.size().width - 2;
				nombre2.append(to_string(num_img) + nombre + "_k" + to_string(num_k) + ".JPG");

				cv::imwrite(nombre2, corregida);
			}
		}
	}
}

void CorrigePezParrot(string ruta_carpeta_entrada, string& banda_extension, string ruta_salida_imagen_corregida)
{	/// CORRECCIÓN DE DISTORSIONES CON FORMULACIÓN DE PARROT PARA CÁMARA MONOCROMÁTICA CON LENTE OJO DE PEZ
	/// **********************************************************************************************************************************************************************
	path ruta(ruta_carpeta_entrada); //"D:/calibracion/"
	int num_img = 0;
	double cx, cy, C, f, p0, p1, p2, p3;
	double Rad1, Rad2, Rad3, Tan1, Tan2;
	bool ojopez = false; // Inicio la variable suponiendo que se corrige la lente RGB

	
	for (directory_entry p : recursive_directory_iterator(ruta)) // iteración en carpetas y subcarpetas de la ruta
	{
		path ruta_archivo = p; // Cada archivo o subcarpeta localizado se utiliza como clase path
		string direc = ruta_archivo.string(); // convertir la ruta de clase path a clase string
		string nombre = ruta_archivo.filename().string(); // convertir el nombre del archivo de clase path a string

		if (nombre.size() > 8 && nombre.compare(nombre.size() - 7, 7, banda_extension) == 0) // Si el nombre es grande de 8 caracteres significa que es una foto (evita carpetas) y si es la banda que quieres.
		{
			num_img += 1; // cada vez que entra significa que encuentra una imagen de la banda que quiere
			
			if (ruta_archivo.extension() == ".TIF")	ojopez = true; // Si la extensión es .TIF es de las cámaras monocromáticas y utiliza una corrección para lente ojo de pez.

			/// LECTURA DE LOS METADATOS DE INTERÉS DE LA IMAGEN ************************************************************************************************************************
			{
				// Abrir la imagen y leer metadatos
				Exiv2::Image::AutoPtr imagen = Exiv2::ImageFactory::open(direc);
				assert(imagen.get() != 0);
				imagen->readMetadata(); // Leer metadatos

				Exiv2::XmpData &XmpData = imagen->xmpData(); // Comprobación por si encuentra los metadatos XMP
				if (XmpData.empty()) {
					std::string error = "No XMP data found in the file";
					throw Exiv2::Error(1, error);
				}

				// Recorrido por todos los campos detectados en los metadatos XMP
				for (Exiv2::XmpData::const_iterator i = XmpData.begin(); i != XmpData.end(); ++i)
				{						
					if (i->key() == "Xmp.Camera.PrincipalPoint")
					{
						string cadena = i->value().toString();
						cx = stod(cadena.substr(0, 8));
						cy = stod(cadena.substr(9, 16));
					}

					if (i->key() == "Xmp.Camera.FisheyeAffineMatrix")
					{
						string cadena = i->value().toString();
						C = stod(cadena.substr(0, 14));
						f = 2 * C / PI;
					}

					if (i->key() == "Xmp.Camera.FisheyePolynomial")
					{
						string cadena = i->value().toString();
						p0 = stod(cadena.substr(0, 1));
						p1 = stod(cadena.substr(2, 3));
						p2 = stod(cadena.substr(4, 15));
						p3 = stod(cadena.substr(16, 27));
					}

					if (i->key() == "Xmp.Camera.PerspectiveFocalLength") // Distancia focal para sensor RGB
					{
						string cadena = i->value().toString();
						f = stod(cadena);
					}

					if (i->key() == "Xmp.Camera.PerspectiveDistortion") // Distancia focal para sensor RGB
					{
						string cadena = i->value().toString();
						Rad1 = stod(cadena.substr(0, 11));
						Rad2 = stod(cadena.substr(12, 24));
						Rad3 = stod(cadena.substr(25, 36));
						Tan1 = stod(cadena.substr(37, 49));
						Tan2 = stod(cadena.substr(50, 60)); 
					}
					
					/// MOSTRAR METADATOS XMP EN PANTALLA DE FORMA COOL
					
					const char* tn = i->typeName();									// typeName() hace referencia al tipo de dato: XmpText
					std::cout << std::setw(44) << std::setfill(' ') << std::left
						<< i->key() << " "											// key() hace referencia al campo que se obtiene
						<< "0x" << std::setw(4) << std::setfill('0') << std::right
						<< std::hex << i->tag() << " "
						<< std::setw(9) << std::setfill(' ') << std::left
						<< (tn ? tn : "Unknown") << " "
						<< std::dec << std::setw(3)
						<< std::setfill(' ') << std::right
						<< i->count() << "  "										// count() hace referencia a la camtidad de datos que se obtiene en el campo. P.E.: Un XmpText con un value()= Sequoia tendrá count()=7.
						<< std::dec << i->value()									// value() hace referencia al valor del metadato para el campo
						<< "\n";
					
				}

				Exiv2::ExifData &exifData = imagen->exifData(); // Comprobación por si encuentra los metadatos Exif
				if (exifData.empty()) {
					std::string error = "No Exif data found in the file";
					throw Exiv2::Error(1, error);
				}

				// Recorrido por todos los campos detectados en los metadatos Exif
				for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != exifData.end(); ++i)
				{
					/// TRANSFORMACIÓN DEL PUNTO PRINCIPAL A PÍXELES PARA LENTE OJO DE PEZ MONOCROMÁTICA
					{
						if (i->key() == "Exif.Image.ImageWidth")
						{
							long numero = i->value().toLong();
							cx = cx * double(numero) / 4.8; // 4.8 mm es el valor del ancho total del sensor monocromático
							// El valor cx se toma en mm y se pasa a píxeles			
						}

						if (i->key() == "Exif.Image.ImageLength")
						{
							long numero = i->value().toLong();
							cy = cy * double(numero) / 3.6; // 3.6 mm es el valor del alto total del sensor monocromático
						}
					}
					/// TRANSFORMACIÓN DEL PUNTO PRINCIPAL A PÍXELES PARA LENTE RGB
					{
						if (i->key() == "Exif.Photo.PixelXDimension")
						{
							long numero = i->value().toLong();
							cx = cx * double(numero) / 6.17472; // 6.17472 mm es el valor del ancho total del sensor RGB
						}

						if (i->key() == "Exif.Photo.PixelYDimension")
						{
							long numero = i->value().toLong();
							cy = cy * double(numero) / 4.63104; // 4.63104 mm es el valor del alto total del sensor RGB
						}
					}

					/// TRANSFORMACIÓN DE LA DISTANCIA FOCAL A PÍXELES PARA LENTE RGB
					{
						if (i->key() == "Exif.Photo.FocalPlaneXResolution")
						{
							auto numero = i->value().toRational();
							double num = numero.first / numero.second;
							f = f * num;
						}
					}

					/// MOSTRAR METADATOS EXIF EN PANTALLA DE FORMA COOL
					
					const char* tn = i->typeName();
					std::cout << std::setw(44) << std::setfill(' ') << std::left
						<< i->key() << " "
						<< "0x" << std::setw(4) << std::setfill('0') << std::right
						<< std::hex << i->tag() << " "
						<< std::setw(9) << std::setfill(' ') << std::left
						<< (tn ? tn : "Unknown") << " "
						<< std::dec << std::setw(3)
						<< std::setfill(' ') << std::right
						<< i->count() << "  "
						<< std::dec << i->value()
						<< "\n";
					
				}
			}
			/// **************************************************************************************************************************************************************
			
			Mat imagen; // imagen que se va a cargar
			Mat map_x, map_y;
			Mat resultado;
			

			if (ojopez)
			{
				imagen = cv::imread(direc, CV_LOAD_IMAGE_UNCHANGED); // Para abrir imágenes de 16 bits por píxel CV_LOAD_IMAGE_ANYDEPTH. Para abrir RGB -> CV_LOAD_IMAGE_COLOR
				imagen.convertTo(imagen, CV_8UC1, 0.0038910505836576); // Divide el nivel digital del píxel entre 257 para hacer la transformación entre 10 bits y 8 bits de profundidad
				map_x.create(imagen.size(), CV_32FC1);
				map_y.create(imagen.size(), CV_32FC1);

				 // Se corrige cada píxel de la imagen
				for (int i = 0; i < imagen.rows; i++)
				{
					for (int j = 0; j < imagen.cols; j++)
					{
						float a = (j - (float)cx) / (float)f; // a y b serán valores que almacenarán la tranformación
						float b = (i - (float)cy) / (float)f;

						double theta = 2.0 / PI * std::atan(sqrt(a*a + b * b));
						double ro = p0 + p1 * theta + p2 * theta*theta + p3 * theta*theta*theta;

						float xd = (float)(C * (ro*a / (sqrt(a*a + b * b))) + cx);
						float yd = (float)(C * (ro*b / (sqrt(a*a + b * b))) + cy);

						map_x.at<float>(i, j) = xd;
						map_y.at<float>(i, j) = yd;
					}
				}
			}
			else
			{
				imagen = cv::imread(direc, CV_LOAD_IMAGE_COLOR);
				map_x.create(imagen.size(), CV_32FC1);
				map_y.create(imagen.size(), CV_32FC1);

				for (int i = 0; i < imagen.rows; i++)
				{
					for (int j = 0; j < imagen.cols; j++)
					{
						float a = (j - (float)cx) / (float)f; // a y b serán valores que almacenarán la tranformación. A para x
						float b = (i - (float)cy) / (float)f;

						float rcuad = a * a + b * b;

						float xd = (float)((1 + Rad1 * rcuad + Rad2 * rcuad*rcuad + Rad3 * rcuad*rcuad*rcuad)*a + 2 * Tan1*a*b + Tan2 * (rcuad + 2 * a*a));
						float yd = (float)((1 + Rad1 * rcuad + Rad2 * rcuad*rcuad + Rad3 * rcuad*rcuad*rcuad)*b + 2 * Tan2*a*b + Tan1 * (rcuad + 2 * b*b));

						xd = f * xd + cx;
						yd = f * yd + cy;

						//if (xd < 0) xd = 0.01;
						//if (xd > 0) xd = imagen.cols-1;
						//if (yd < 0) yd = 0.01;
						//if (yd > 0) yd = imagen.rows-1;
						map_x.at<float>(i, j) = xd;
						map_y.at<float>(i, j) = yd;
					}
				}
			}

			cv::remap(imagen, resultado, map_x, map_y, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0)); // Mirar parámetros en profundidad. Conveniencia.
			
			string salida = ruta_salida_imagen_corregida;
			salida.append(to_string(num_img) + banda_extension);
			cv::imwrite(salida, resultado);
		}
	}
	
	/// **********************************************************************************************************************************************************************
	/// **********************************************************************************************************************************************************************
}