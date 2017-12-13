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
	imshow("Image 1", img_base);
	imshow("Image 2", img_movida);
	imshow("Image 2 Alineada", img_alineada);
	waitKey(0);
	
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

					cvtColor(imagen, imagen, CV_BGR2GRAY);
					
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


void CalibraMonoOjoPez(string ruta_carpeta_entrada, string& banda_extension, int& num_k, string ruta_salida_deteccionesquina)
{
	/// CALIBRACIÓN DE UN SET DE IMÁGENES PARA UNA BANDA MONOCROMÁTICA
	/// **********************************************************************************************************************************************************************
	
	Mat imagen=imread("D:/calibracion/0331/IMG_700101_000101_0000_GRE.TIF", CV_LOAD_IMAGE_UNCHANGED);
	imagen.convertTo(imagen, CV_8UC1, 0.0038910505836576);
	Mat map_x, map_y;
	Mat resultado;
	map_x.create(imagen.size(), CV_32FC1);
	map_y.create(imagen.size(), CV_32FC1);

	for (int i = 0; i < imagen.rows; i++)
	{
		for (int j = 0; j < imagen.cols; j++)
		{
			//map_x.at<float>(i, j)
			//map_y.at<float>(i, j)
			double cx = 1280 * 2.312299 / 4.8;
			double cy = 960 * 1.780116 / 3.6;
			double f = 1060.042371;
			double C = 1665.110663867;
			
			float a = (j - cx) / f;
			float b = (i - cy) / f;
				
			double theta = 2.0 / 3.141592 * std::atan(sqrt(a*a + b*b));
			double ro = theta + 0.011488602 * theta*theta - 0.14704581 *theta*theta*theta;

			double xh = C*(ro*a / (sqrt(a*a + b*b))) + cx;
			double yh = C*(ro*b / (sqrt(a*a + b*b))) + cy;
			
			map_x.at<float>(i,j) = xh;
			map_y.at<float>(i,j) = yh;
		}
	}
	
	
	GuardarMat(map_x, "COSA.yml", "map_x");
	GuardarMat(map_y, "COSA2.yml", "map_y");

	remap(imagen, resultado, map_x, map_y, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));

	imshow("HOLA", resultado);
	waitKey(0);










	path ruta(ruta_carpeta_entrada); //"D:/calibracion/"
	int num_img = 0;
	vector<vector<cv::Point2f>> coord_img;
	vector<vector<cv::Point3f>> coord_obj; // Vector de vectores de puntos en coordenadas locales (se repiten siempre en cada imagen)
	cv::Mat matrizcampez = Mat::zeros(Size(3, 3), CV_64F); // Matriz intrínseca de la cámara
	cv::Mat distcoefpez = Mat::zeros(Size(1,4), CV_64F); // Matriz de coeficientes de distorsión de la cámara
	
	/*
	matrizcampez.at<double>(0,0) = 1060.042371;
	matrizcampez.at<double>(1, 1) = 1060.042371;
	matrizcampez.at<double>(2, 2) = 1.0;
	matrizcampez.at<double>(0,2) = 1280*2.312299/4.8;
	matrizcampez.at<double>(1,2) = 1.780116*960/3.6;

	distcoefpez.at<double>(0, 0) = 0.011488602;
	distcoefpez.at<double>(1,0) = -0.14704581;
	*/

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

				bool loc = findChessboardCorners(imagen, tamano, esquinas, CV_CALIB_CB_ADAPTIVE_THRESH);
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
					cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Se transforma la imagen a 3 canales de color para ponerle la detección de esquinas en color.

					cv::drawChessboardCorners(imagen, tamano, esquinas, loc); // Dibujo de las esquinas detectadas en colores

					cv::imwrite(ruta_salida_deteccionesquina + std::to_string(num_img) + banda_extension, imagen);
				}

			}
		}

	}
	
	vector<vector<cv::Point2f>> coord_img_trans;
	for (int i = 0; i < coord_img.size(); i++)
	{
		vector<Point2f> lista_temp;
		for (int j = 0; j < coord_img[i].size(); j++)
		{
			Point2f temp;
			temp.x=(coord_img[i][j].x - (1280 * 2.312299 / 4.8)) / 1060.042371;
			temp.y= (coord_img[i][j].y - (1.780116 * 960 / 3.6)) / 1060.042371;
			double theta = 2.0 / 3.141592 * std::atan(sqrt(temp.x*temp.x + temp.y*temp.y));
			double ro = theta + 0.011488602 * theta*theta - 0.14704581 *theta*theta*theta;
			Point2f Sin_dist;
			Sin_dist.x = 1665.110663867*(ro*temp.x / (sqrt(temp.x*temp.x + temp.y*temp.y))) + (1280 * 2.312299 / 4.8);
			Sin_dist.y= 1665.110663867*(ro*temp.y / (sqrt(temp.x*temp.x + temp.y*temp.y))) + (1.780116 * 960 / 3.6);
			lista_temp.push_back(Sin_dist);
		}
		coord_img_trans.push_back(lista_temp);
	}

	
	/// POSBILES CALIBRACIONES SEGÚN EL NÚMERO DE PARÁMETROS DE DISTORSIÓN RADIAL QUE SE QUIERAN UTILIZAR
	double error_repro;
	if (num_k == 3)
	{
		error_repro = fisheye::calibrate(coord_obj, coord_img, Size(1280, 960), matrizcampez, distcoefpez, rotmat, trasmat, 
			fisheye::CALIB_RECOMPUTE_EXTRINSIC + /*fisheye::CALIB_CHECK_COND*/ + fisheye::CALIB_FIX_SKEW + fisheye::CALIB_USE_INTRINSIC_GUESS, cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.000001)); // Calibración de la cámara
	}
	//+ fisheye::CALIB_USE_INTRINSIC_GUESS+fisheye::CALIB_FIX_K3 + fisheye::CALIB_FIX_K4
	



	double a = distcoefpez.at<double>(0, 0);
	double b = distcoefpez.at<double>(1,0);
	double c = distcoefpez.at<double>(2,0);
	double d = distcoefpez.at<double>(3,0);

	a=matrizcampez.at<double>(0, 0);
	b=matrizcampez.at<double>(1, 1);
	c=matrizcampez.at<double>(0, 2);
	d=matrizcampez.at<double>(1, 2);
	a=matrizcampez.at<double>(2, 2);


	Mat imagenmala = imread("D:/calibracion/0331/IMG_700101_000101_0000_GRE.TIF", CV_LOAD_IMAGE_UNCHANGED);
	imagenmala.convertTo(imagenmala, CV_8UC1, 0.0038910505836576); // Le pongo un canal para el gris. El factor de escala (1/257) reduce el 65535 máximo valor en profundidad de 16 bits a un ND de 255.
	Mat imagenbuena;
	
	fisheye::undistortImage(imagenmala, imagenbuena, matrizcampez, distcoefpez);

	imshow("COSA", imagenbuena);
	waitKey(0);
	
	
	
	
	
	double ancho = 4.8; // parámetros obtenidos mediante la multiplicación del tamaño del píxel (3.75 micras para monocromáticas) y de la resolución de la imagen 
	double alto = 3.6; // especifican el ancho y el alto del tamaño del sensor. Probablemente por el tipo de obturador CCD mono y CMOS RGB.
	double fov_x;
	double fov_y;
	double dist_focal;
	cv::Point2d punto_prin;
	double ratio_aspecto;

	//cv::calibrationMatrixValues(matrizcampez, Size(1280, 960), ancho, alto, fov_x, fov_y, dist_focal, punto_prin, ratio_aspecto); // Parámetros físicos de la cámara

	/// OBTENCIÓN DE LAS MATRICES DE CALIBRACIÓN
	/*
	vector<double>matrizfisicapez; // Vector para obtener la salida de parámetros
	matrizfisicapez.push_back(dist_focal);
	matrizfisicapez.push_back(ratio_aspecto);
	matrizfisicapez.push_back(punto_prin.x);
	matrizfisicapez.push_back(punto_prin.y);
	matrizfisicapez.push_back(fov_x);
	matrizfisicapez.push_back(fov_y);
	matrizfisicapez.push_back(error_repro);
	*/

	string nombre_salida = banda_extension;
	for (int i = 0; i <= 3; i++) // quitarle el .TIF a las bandas
	{
		nombre_salida.pop_back();
	}

	/*
	string nombre_sal_puntos = nombre_salida;
	nombre_sal_puntos.append(".yml");
	string salida = "Listado_Puntos_";
	salida.append(nombre_sal_puntos);
	GuardarPuntosDetectados(coord_img, salida); // salida = Listado_Puntos_GRE.yml
	*/

	nombre_salida.append("_k");
	nombre_salida.append(to_string(num_k));
	nombre_salida.append(".yml");

	//salida = "Matriz_Fisica_";
	//salida.append(nombre_salida);
	//GuardarMatFisica(matrizfisicapez, salida); // salida = Matriz_Fisica_GRE_k3.yml

	string salida = "Matriz_Camara_PEZ_";
	salida.append(nombre_salida);
	GuardarMatCamara(matrizcampez, salida);

	salida = "Matriz_Distorsion_PEZ_";
	salida.append(nombre_salida);
	GuardarMatDistorsion(distcoefpez, salida);

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