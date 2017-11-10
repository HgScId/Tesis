#include <opencv2/opencv.hpp>
#include <iostream>
#include <conio.h>
#include <stdio.h>


#include <opencv2\reg\map.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>



///////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;

int main() {
	cv::Mat img1 = cv::imread("calibracion/13RojoC.TIF", CV_LOAD_IMAGE_UNCHANGED);
	img1.depth();
	img1 /= 255;
	img1.convertTo(img1, CV_8UC1); // Le pongo un canal para el gris.
	cv::Mat img2 = cv::imread("calibracion/13VerdeC.TIF", CV_LOAD_IMAGE_UNCHANGED);
	img2.depth();
	img2 /= 255;
	img2.convertTo(img2, CV_8UC1); // Le pongo un canal para el gris.

	vector<cv::Point2f> esquinas2;
	cv::Size tamano2(8, 6);
	vector<cv::Point2f> esquinas3;
	cv::Size tamano3(8, 6);

	bool loc2 = cv::findChessboardCorners(img1, tamano2, esquinas2, CV_CALIB_CB_ADAPTIVE_THRESH);
	cv::cornerSubPix(img1, esquinas2, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

	bool loc3 = cv::findChessboardCorners(img1, tamano3, esquinas3, CV_CALIB_CB_ADAPTIVE_THRESH);
	cv::cornerSubPix(img1, esquinas2, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

	cv::Mat homografia = cv::findHomography(esquinas2, esquinas3);

	for (int i = 0; i <= 2; i++)
	{
		for (int j = 0; j <= 2; j++)
		{
			std::cout << homografia.at<double>(i, j) << " ";
		}
		std::cout << endl;
	}

	cv::Mat alineada(cv::Size(1300, 980), CV_8UC1);
	cv::warpPerspective(img2, alineada, homografia, alineada.size());
	cv::imwrite("pruebaalinea.TIF", alineada);

	cv::Mat imagenfin(cv::Size(1280, 960), CV_8UC3);
	//cv::reg
	cv::Mat g = cv::Mat::zeros(cv::Size(img1.cols, img1.rows), CV_8UC1);
	vector<cv::Mat> canales;
	canales.push_back(alineada);
	canales.push_back(img1);
	canales.push_back(g);
	cv::merge(canales, imagenfin);
	cv::imshow("img", imagenfin);
	cv::waitKey(0);




	// Vectores de vectores de puntos. Estructura que utiliza la función de calibración de la cámara
	vector<vector<cv::Point3f>> coord_obj;
	vector<vector<cv::Point2f>> coord_imagen;

	cv::Mat imagen;
	cv::Size tamano(8, 6);

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
		nombre.append("Rojo.TIF"); // nombre variable

		imagen = cv::imread(nombre, CV_LOAD_IMAGE_UNCHANGED); // Para abrir imágenes de 16 bits por píxel CV_LOAD_IMAGE_ANYDEPTH. Para abrir RGB -> CV_LOAD_IMAGE_COLOR
															  // Las imágenes tienen una profundidad de 16 bits por píxel y se cargan con CV_LOAD_IMAGE_UNCHANGED / CV_LOAD_IMAGE_ANYDEPTH. 
															  // Es decir 2^16 (0 - 65535) ND, 65536 niveles digitales. Por eso CorelDraw da problemas. depth devuelve 2 al ser CV_16
															  // La imagen tiene sólo un canal.


															  // int imgDepth = imagen.depth(); Función que devuelve los bits de la imagen por píxel.
															  // imagen.channels(); Función que devuelve los canales de una imagen.
															  // Imágenes de 1280 x 960
															  // RGB en 4608 x 3456 y en 8 bits por canal y píxel de profundidad

															  // Conversión de la imagen a 8 bits
		imagen /= 255;
		imagen.convertTo(imagen, CV_8UC1); // Le pongo un canal para el gris.


										   // Localización de las esquinas del tablero.
		bool loc = cv::findChessboardCorners(imagen, tamano, esquinas, CV_CALIB_CB_ADAPTIVE_THRESH);
		cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		// Mejora la detección de las esquinas a escala subpixel. 
		// Parece no tolerar la entrada de valores en double CV_64F, sólo float CV_32F.

		coord_imagen.push_back(esquinas); // introduces las esquinas en el vector general

										  // Inclusión de 3 canales RGB para el dibujo de las esquinas con color.
		cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Para detectar las esquinas en color

												   // Dibujo de las esquinas detectadas en colores
		cv::drawChessboardCorners(imagen, tamano, esquinas, loc);

		cv::imwrite("deteccionesquina//" + std::to_string(i) + "Rojo.TIF", imagen);

		//cv::namedWindow("hola", CV_WINDOW_FULLSCREEN);
		//cv::imshow("hola", imagen);
		//cv::waitKey(0);
	}

	cv::Mat matrizcam(cv::Size(3, 3), CV_64F);
	//matrizcam.at<double>(1, 1) = 1.0; No condiciona nada fijar los valores de partida
	//matrizcam.at<double>(0, 0) = 1.0;

	cv::Mat distcoef(cv::Size(5, 1), CV_64F);
	vector<cv::Mat> rotmat;
	vector<cv::Mat> trasmat;

	// | CV_CALIB_ZERO_TANGENT_DIST Si se quiere anular la distorsión tangencial... es un parámetro pequeño y se supone que el sensor y la lente están bien alineadas en cámaras de calidad.
	// | CV_CALIB_FIX_K3 Para fijar k3=0
	// CV_CALIB_FIX_ASPECT_RATIO fx y fy se fijan iguales
	// La combinación que mejores resultados da es sin poner ninguna flag al algoritmo.
	cv::calibrateCamera(coord_obj, coord_imagen, imagen.size(), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_FIX_K3);

	// cv::Mat matrizcamoptima(cv::Size(3, 3), CV_64F);
	// matrizcamoptima = cv::getOptimalNewCameraMatrix(matrizcam,distcoef,imagen.size(),1); // sirve para conservar todos los píxeles de la imagen original
	// Para utilizarlo se le añade a la función undistort como último parámetro.

	for (int i = 0; i <= 2; i++)
	{
		for (int j = 0; j <= 2; j++)
		{
			std::cout << matrizcam.at<double>(i, j) << " ";
		}
		std::cout << endl;
	}

	for (int j = 0; j <= 4; j++)
	{
		std::cout << distcoef.at<double>(0, j) << " ";
	}
	std::cout << endl;

	for (int i = 1; i <= 18; i++)
	{
		std::string nombre = "calibracion/";
		nombre.append(std::to_string(i));
		nombre.append("Rojo.TIF"); // nombre variable
		cv::Mat corregir = cv::imread(nombre, CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
		cv::Mat corregida;
		cv::undistort(corregir, corregida, matrizcam, distcoef);
		std::string nombre2 = "calibracion/";
		nombre2.append(std::to_string(i));
		nombre2.append("RojoC.TIF"); // nombre variable
		cv::imwrite(nombre2, corregida);
	}

	cv::Mat prueba = cv::imread("pruebaREG.TIF", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
	cv::Mat pruebabien;
	cv::undistort(prueba, pruebabien, matrizcam, distcoef);
	cv::imwrite("pruebaREGbien.TIF", pruebabien);

	cv::imshow("prueba", prueba);
	cv::imshow("pruebabien", pruebabien);
	cv::waitKey(0);

	// Calcular el error de reproyección
	double error = 0.0;
	for (int i = 0; i <= 17; i++)
	{
		vector<cv::Point2f> coord_obj_proy;
		cv::projectPoints(coord_obj[i], rotmat[i], trasmat[i], matrizcam, distcoef, coord_obj_proy);
		error += cv::norm(coord_imagen[i], coord_obj_proy, cv::NORM_L2); // acumula en error la raíz de los cuadrados de los módulos de los vectores entre la esquina detectada y la reproyectada
	}
	double RMSE = error / sqrt(48 * 18);

	// Obtener parámetros reales de la cámara

	double ancho = 4.8; // parámetros obtenidos mediante la multiplicación del tamaño del píxel (3.75 micras para monocromáticas) y de la resolución de la imagen 
	double alto = 3.6; // especifican el ancho y el alto del tamaño del sensor. Probablemente por el tipo de obturador CCD mono y CMOS RGB.
	double fovx;
	double fovy;
	double focallength;
	cv::Point2d PrincipalPoint;
	double AspectRatio;

	cv::calibrationMatrixValues(matrizcam, imagen.size(), ancho, alto, fovx, fovy, focallength, PrincipalPoint, AspectRatio);


	return(0);


}

/*

/// LOCALIZAR PIXEL EN IMAGEN ORGINAL. SE TRATA DE UNA MATRIZ DE UNSIGNED SHORT:
auto a = imagen.at<unsigned short>(1, 1);

TRAS LA CONVERSION A 8 BITS Y AÑADIRLE TRES CANALES:
imagen.at<cv::Vec3b>(1, 1)[0];
imagen.at<cv::Vec3b>(1, 1)[1];
imagen.at<cv::Vec3b>(1, 1)[2];


/// GUARDAR MATRICES EN ARCHIVO DE TEXTO:
cv::FileStorage guardar("cosa.yml", cv::FileStorage::WRITE);
guardar << "matriz inicial" << imagen;
guardar.release();

/// LOCALIZAR MIN Y MAX DE LA MATRIZ
double min, max;
cv::minMaxLoc(imagen, &min, &max);
*/