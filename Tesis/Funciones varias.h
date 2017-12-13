#pragma once
#include <opencv2/core/core.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cv;
using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////


/// ALINEAR IM�GENES CALIBRADAS
// Se utiliza para image registration con traslaci�n. Im�genes en .TIF y 10 (16) bits.
Mat AlineaImg(Mat& imgbase, Mat& imgmovida);

/// GUARDAR MATRIZ/VECTOR/IMAGEN CUALQUIERA EN ARCHIVO DE TEXTO
// Guardar matriz de forma b�sica con los campos por defecto. Formato .yml
// ruta es la ruta con el nombre de archivo y el formato incluido.
// nombre_archivo es el nombre de la matriz dentro del archivo.
void GuardarMat(Mat& matriz, string ruta, string nombre);

void GuardarMatFisica(vector<double> & vector, string ruta);
void GuardarMatDistorsion(Mat& matriz, string ruta);
void GuardarMatCamara(Mat& matriz, string ruta);

void GuardarPuntosDetectados(vector<vector<Point2f>> & coord_img, string ruta);



/// LEER MATRICES INTR�NSECAS DE LA C�MARA Y ALMACENARLAS EN UNA MATRIZ
Mat LeerMatCamara(string ruta_ext);
Mat LeerMatDistorsion(string ruta_ext, int max_k);

vector<vector<Point2f>> LeerPuntosDetectados(string ruta, int num_img);

/// CALIBRA IM�GENES Y SE OBTIENE LA DETECCI�N DE ESQUINAS. Realizado para un tablero de 12x8 esquinas con cuadrados de 30x30 cm.
void CalibraRGB(string ruta_carpeta_entrada, string& banda_extension, int& num_k, string ruta_salida_deteccionesquina);
void CalibraMono(string ruta_carpeta, string& banda_extension, int& num_k, string ruta_salida_deteccionesquina);

void CalibraMonoOjoPez(string ruta_carpeta_entrada, string& banda_extension, int& num_k, string ruta_salida_deteccionesquina);



/// CORRECCI�N DE SET DE IM�GENES A PARTIR DE LAS MATRICES INTR�NSECAS DE LA C�MARA
void CorrigeImagenes(Mat& mat_cam, Mat& dist_coef, string& banda, string ruta_img_entrada, string ruta_img_salida);
void CorrigeImagenesRGB(Mat& mat_cam, Mat& dist_coef, string& banda, string ruta_img_entrada, string ruta_img_salida);