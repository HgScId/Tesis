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



using namespace cv;
using namespace std;

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

void GuardarMat(Mat& matriz, string ruta, string nombre_archivo)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	guardar << nombre_archivo << matriz;
	guardar.release();
}

void GuardarMat(vector<double> & vector, string ruta, string nombre_archivo)
{
	cv::FileStorage guardar(ruta, cv::FileStorage::WRITE);
	guardar << nombre_archivo << vector;
	guardar.release();
}

void CalibraRGB()
{
	/// CALIBRACIÓN DE UN SET DE IMÁGENES PARA BANDA RGB
	/// **********************************************************************************************************************************************************************

	path ruta("D:/calibracion/");
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
			if (nombre.compare(nombre.size() - 7, 7, "RGB.JPG") == 0) // Encuentra imagen banda buscada
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
					coord_imagen.push_back(esquinas); // introduces las esquinas detectadas en coordenadas de la imagen en el vector general

					cvtColor(imagen, imagen, CV_BGR2GRAY);
					
					cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1)); // función para la detección precisa de las esquinas
					
					cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Se transforma la imagen a 3 canales de color para ponerle la detección de esquinas en color.
					cv::drawChessboardCorners(imagen, tamano, esquinas, loc); // Dibujo de las esquinas detectadas en colores
					

					cv::imwrite("deteccionesquina//" + std::to_string(num_img) + "RGB.JPG", imagen);
				}

			}
		}

	}

	double cal = cv::calibrateCamera(coord_obj, coord_imagen, Size(4608,3456), matrizcam, distcoef, rotmat, trasmat); // Calibración de la cámara
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
	matrizfisica.push_back(cal);

	GuardarMat(matrizfisica, "Matriz_Fisica.yml", "Matriz Fisica");

	GuardarMat(matrizcam, "Matriz_Camara.yml", "Matriz Camara");

	GuardarMat(distcoef, "Matriz_Distorsion.yml", "Matriz Distorsion");


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
	/// **********************************************************************************************************************************************************************
	/// **********************************************************************************************************************************************************************
}
