#include <cstdio> // Para utilizar archivos: fopen, fread, fwrite, gets...
#include <iostream> // Funciones cout y cin



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

int main() {
	/*
	Mat im1 = imread("pruebaRbienk61.TIF", CV_LOAD_IMAGE_UNCHANGED);
	im1.at<unsigned short>(0, 1189) = 3;
	Mat im2 = imread("pruebaVbienk61.TIF", CV_LOAD_IMAGE_UNCHANGED);
	Mat resultado = AlineaImg(im1, im2);
	imwrite("alineaRVk61.TIF", resultado);
	*/
	/*
	Mat im1 = imread("calibracion/13RojoC.TIF", CV_LOAD_IMAGE_UNCHANGED);
	im1.at<unsigned short>(0, 1189) = 3;
	Mat im2 = imread("calibracion/13VerdeC.TIF", CV_LOAD_IMAGE_UNCHANGED);
	Mat resultado = AlineaImg(im1, im2);
	imwrite("alinea13.TIF", resultado);
	*/

	/*
	// Read points

	std::vector<cv::Point2f> imagePoints = Generate2DPoints();

	std::vector<cv::Point3f> objectPoints = Generate3DPoints();



	std::cout << "There are " << imagePoints.size() << " imagePoints and " << objectPoints.size() << " objectPoints." << std::endl;

	cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);

	cv::setIdentity(cameraMatrix);



	std::cout << "Initial cameraMatrix: " << cameraMatrix << std::endl;



	cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);

	distCoeffs.at<double>(0) = 0;

	distCoeffs.at<double>(1) = 0;

	distCoeffs.at<double>(2) = 0;

	distCoeffs.at<double>(3) = 0;



	cv::Mat rvec(3, 1, cv::DataType<double>::type);

	cv::Mat tvec(3, 1, cv::DataType<double>::type);



	cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);



	std::cout << "rvec: " << rvec << std::endl;

	std::cout << "tvec: " << tvec << std::endl;



	std::vector<cv::Point2f> projectedPoints;

	cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);



	for (unsigned int i = 0; i < projectedPoints.size(); ++i)

	{

		std::cout << "Image point: " << imagePoints[i] << " Projected to " << projectedPoints[i] << std::endl;

	}

	*/


	// Vectores de vectores de puntos. Estructura que utiliza la función de calibración de la cámara
	vector<vector<cv::Point3f>> coord_obj;
	vector<vector<cv::Point2f>> coord_imagen;

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
		nombre.append("REG.TIF"); // nombre variable

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
		cv::cornerSubPix(imagen, esquinas, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		// Mejora la detección de las esquinas a escala subpixel. 
		// Parece no tolerar la entrada de valores en double CV_64F, sólo float CV_32F.

		coord_imagen.push_back(esquinas); // introduces las esquinas en el vector general

		// Inclusión de 3 canales RGB para el dibujo de las esquinas con color.
		cv::cvtColor(imagen, imagen, CV_GRAY2RGB); // Para detectar las esquinas en color

		// Dibujo de las esquinas detectadas en colores
		cv::drawChessboardCorners(imagen, tamano, esquinas, loc);

		cv::imwrite("deteccionesquina//" + std::to_string(i) + "REG.TIF", imagen);

		//cv::namedWindow("hola", CV_WINDOW_FULLSCREEN);
		//cv::imshow("hola", imagen);
		//cv::waitKey(0);
	}
	
	//cv::Mat matrizcam= Mat::eye(3, 3, CV_64F);
	cv::Mat matrizcam = Mat(Size(3, 3), CV_64F);
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
	cv::Mat distcoef = Mat(Size(8, 1), CV_64F);
	distcoef.at<double>(0, 0) = 308.34805117438947;
	distcoef.at<double>(0, 1) = 390.52229314295465;
	distcoef.at<double>(0, 2) = 0.00068590874232127091;
	distcoef.at<double>(0, 3) = -0.00012588044457764470;
	distcoef.at<double>(0, 4) = -315.59251801932652;
	distcoef.at<double>(0, 5) = 308.39330121634094;
	distcoef.at<double>(0, 6) = 507.33006137507221;
	distcoef.at<double>(0, 7) = -290.37421176928223;


	vector<cv::Mat> rotmat;
	vector<cv::Mat> trasmat;
	//Mat rotmat;
	//Mat trasmat;

	// CV_CALIB_ZERO_TANGENT_DIST Si se quiere anular la distorsión tangencial... es un parámetro pequeño y se supone que el sensor y la lente están bien alineadas en cámaras de calidad.
	// CV_CALIB_FIX_K3 Para fijar k3=0
	// CV_CALIB_FIX_ASPECT_RATIO fx y fy se fijan iguales
	// CV_CALIB_RATIONAL_MODEL para usar componentes de k4, k5 y k6
	// CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5| CV_CALIB_FIX_K6
	// CV_CALIB_USE_INTRINSIC_GUESS
	
	double cal = cv::calibrateCamera(coord_obj, coord_imagen, imagen.size(), matrizcam, distcoef, rotmat, trasmat, CV_CALIB_RATIONAL_MODEL);

	
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

	GuardarMat(distcoef, "matriz_distorsion.yml", "Matriz Distorsion");
	/*
	for (int i = 1; i <= 18; i++)
	{
		std::string nombre = "calibracion/";
		nombre.append(std::to_string(i));
		nombre.append("Rojo.TIF"); // nombre variable
		cv::Mat corregir = cv::imread(nombre, CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
		//cv::imshow("prueba", corregir);
		//cv::waitKey(0);
		cv::Mat corregida;
		cv::undistort(corregir, corregida, matrizcam, distcoef);
		//cv::imshow("prueba", corregida);
		//cv::waitKey(0);
		std::string nombre2 = "calibracion/";
		nombre2.append(std::to_string(i));
		nombre2.append("RojoCMV.TIF"); // nombre variable
		cv::imwrite(nombre2, corregida);
	}
	*/
	for (int i = 1; i <= 18; i++)
	{
		std::string nombre = "calibracion/";
		nombre.append(std::to_string(i));
		nombre.append("REG.TIF"); // nombre variable
		cv::Mat corregir = cv::imread(nombre, CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
		//cv::imshow("prueba", corregir);
		//cv::waitKey(0);
		cv::Mat corregida;
		cv::undistort(corregir, corregida, matrizcam, distcoef);
		//cv::imshow("prueba", corregida);
		//cv::waitKey(0);
		std::string nombre2 = "calibracion/";
		nombre2.append(std::to_string(i));
		nombre2.append("NREG.TIF"); // nombre variable
		cv::imwrite(nombre2, corregida);
	}

	cv::Mat prueba = cv::imread("pruebaREG.TIF", CV_LOAD_IMAGE_ANYDEPTH); // Para abrir imágenes de 16 bits por píxel
	cv::Mat pruebabien;
	cv::undistort(prueba, pruebabien, matrizcam, distcoef);
	cv::imwrite("pruebaREGbien.TIF", pruebabien);

	//cv::imshow("prueba", prueba);
	//cv::imshow("pruebabien", pruebabien);
	//cv::waitKey(0);

	
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
/// LOCALIZAR MIN Y MAX DE LA MATRIZ
double min, max;
cv::minMaxLoc(imagen, &min, &max);
*/