#include "Funciones varias.h"


#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>



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
