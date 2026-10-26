#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const string DOSSIER = "images/";
const string NOMS_DET[3] = {"Sobel", "Laplace", "Canny"};

struct Params {
    int detecteur = 2;
    int flou = 4;
    int seuil = 80;
    int seuilBas = 40;
    int seuilHaut = 80;
    int noyau = 0;
    int voisinage = 1;
};

struct Mesures {
    int detectes = 0, reference = 0, corrects = 0, fauxPos = 0, fauxNeg = 0;
    double P = 0, TFP = 0, TFN = 0;
    double precision = 0, rappel = 0, F = 0;
};

vector<string> noms;
vector<Mat> images, references;
Params p;
int indice = 0;

void chargerImages()
{
    vector<String> fichiers;
    glob(DOSSIER + "*.pgm", fichiers);

    for (auto& f : fichiers) {
        string nom = f.substr(f.find_last_of('/') + 1);
        nom = nom.substr(0, nom.size() - 4);

        Mat im = imread(f, IMREAD_GRAYSCALE);
        Mat gt = imread(DOSSIER + "gt/" + nom + "_gt_binary.pgm", IMREAD_GRAYSCALE);
        if (im.empty() || gt.empty()) {
            cout << "Impossible de lire " << nom << endl;
            continue;
        }

        noms.push_back(nom);
        images.push_back(im);
        references.push_back(gt < 128);
    }
    cout << images.size() << " images chargees" << endl;
}

Mat lisser(const Mat& im, int flou)
{
    Mat res;
    im.convertTo(res, CV_32F);
    if (flou > 0)
        GaussianBlur(res, res, Size(0, 0), flou * 0.5);
    return res;
}

Mat sobel(const Mat& im, const Params& q)
{
    Mat gx, gy, mag;
    int k = 2 * q.noyau + 3;
    Mat l = lisser(im, q.flou);
    Sobel(l, gx, CV_32F, 1, 0, k);
    Sobel(l, gy, CV_32F, 0, 1, k);
    magnitude(gx, gy, mag);

    double maxi;
    minMaxLoc(mag, 0, &maxi);
    mag = mag * (255.0 / maxi);

    return mag > q.seuil;
}

Mat laplace(const Mat& im, const Params& q)
{
    Mat lap;
    int k = 2 * q.noyau + 3;
    Laplacian(lisser(im, q.flou), lap, CV_32F, k);

    double mini, maxi;
    minMaxLoc(lap, &mini, &maxi);
    lap = lap * (255.0 / max(fabs(mini), fabs(maxi)));

    Mat res = Mat::zeros(im.size(), CV_8U);
    for (int y = 0; y < lap.rows - 1; y++) {
        for (int x = 0; x < lap.cols - 1; x++) {
            float v = lap.at<float>(y, x);
            float d = lap.at<float>(y, x + 1);
            float b = lap.at<float>(y + 1, x);
            if ((v * d < 0 && fabs(v - d) > q.seuil) || (v * b < 0 && fabs(v - b) > q.seuil))
                res.at<uchar>(y, x) = 255;
        }
    }
    return res;
}

Mat canny(const Mat& im, const Params& q)
{
    Mat l, res;
    lisser(im, q.flou).convertTo(l, CV_8U);
    Canny(l, res, q.seuilBas, max(q.seuilBas, q.seuilHaut), 3, true);
    return res;
}

Mat detecter(const Mat& im, const Params& q)
{
    if (q.detecteur == 0) return sobel(im, q);
    if (q.detecteur == 1) return laplace(im, q);
    return canny(im, q);
}

Mesures comparer(const Mat& det, const Mat& ref, int r)
{
    Mat refDil = ref.clone(), detDil = det.clone();
    if (r > 0) {
        Mat k = getStructuringElement(MORPH_RECT, Size(2 * r + 1, 2 * r + 1));
        dilate(ref, refDil, k);
        dilate(det, detDil, k);
    }

    Mesures m;
    m.detectes = countNonZero(det);
    m.reference = countNonZero(ref);

    Mat tmp;
    bitwise_and(det, refDil, tmp);
    m.corrects = countNonZero(tmp);
    m.fauxPos = m.detectes - m.corrects;

    bitwise_and(ref, ~detDil, tmp);
    m.fauxNeg = countNonZero(tmp);

    double total = m.corrects + m.fauxPos + m.fauxNeg;
    if (total > 0) {
        m.P = m.corrects / total;
        m.TFP = m.fauxPos / total;
        m.TFN = m.fauxNeg / total;
    }
    if (m.detectes > 0) m.precision = (double)m.corrects / m.detectes;
    if (m.reference > 0) m.rappel = (double)(m.reference - m.fauxNeg) / m.reference;
    if (m.precision + m.rappel > 0) m.F = 2 * m.precision * m.rappel / (m.precision + m.rappel);
    return m;
}

Mat imageComparaison(const Mat& im, const Mat& det, const Mat& ref, int r)
{
    Mat refDil = ref.clone(), detDil = det.clone();
    if (r > 0) {
        Mat k = getStructuringElement(MORPH_RECT, Size(2 * r + 1, 2 * r + 1));
        dilate(ref, refDil, k);
        dilate(det, detDil, k);
    }

    Mat res;
    cvtColor(im / 3 + 170, res, COLOR_GRAY2BGR);
    res.setTo(Scalar(255, 0, 0), ref & ~detDil);
    res.setTo(Scalar(0, 0, 255), det & ~refDil);
    res.setTo(Scalar(0, 170, 0), det & refDil);
    return res;
}

Mesures moyenne(const vector<Mesures>& liste)
{
    Mesures m;
    for (auto& x : liste) {
        m.P += x.P;
        m.TFP += x.TFP;
        m.TFN += x.TFN;
        m.precision += x.precision;
        m.rappel += x.rappel;
        m.F += x.F;
    }
    m.P /= liste.size();
    m.TFP /= liste.size();
    m.TFN /= liste.size();
    m.precision /= liste.size();
    m.rappel /= liste.size();
    m.F /= liste.size();
    return m;
}

vector<Mesures> traiterLot(const Params& q, const string& csv = "", const string& dossierSortie = "")
{
    vector<Mesures> res;
    filesystem::create_directories("resultats");
    if (!dossierSortie.empty())
        filesystem::create_directories(dossierSortie);
    ofstream f;
    if (!csv.empty()) {
        f.open(csv);
        f << "image;contours_detectes;contours_reference;contours_corrects;faux_positifs;faux_negatifs;P;TFP;TFN;precision;rappel;F\n";
    }

    for (size_t i = 0; i < images.size(); i++) {
        Mat det = detecter(images[i], q);
        Mesures m = comparer(det, references[i], q.voisinage);
        res.push_back(m);

        if (f.is_open())
            f << noms[i] << ";" << m.detectes << ";" << m.reference << ";" << m.corrects << ";"
              << m.fauxPos << ";" << m.fauxNeg << ";" << m.P << ";" << m.TFP << ";" << m.TFN << ";" << m.precision << ";" << m.rappel << ";" << m.F << "\n";

        if (!dossierSortie.empty()) {
            imwrite(dossierSortie + noms[i] + "_contours.png", 255 - det);
            imwrite(dossierSortie + noms[i] + "_comparaison.png", imageComparaison(images[i], det, references[i], q.voisinage));
        }
    }

    Mesures moy = moyenne(res);
    if (f.is_open())
        f << "MOYENNE;;;;;;" << moy.P << ";" << moy.TFP << ";" << moy.TFN << ";" << moy.precision << ";" << moy.rappel << ";" << moy.F << "\n";
    return res;
}

void afficher(int, void*)
{
    const Mat& im = images[indice];
    Mat det = detecter(im, p);
    Mesures m = comparer(det, references[indice], p.voisinage);

    Mat a, b, c, d;
    cvtColor(im, a, COLOR_GRAY2BGR);
    cvtColor(255 - det, b, COLOR_GRAY2BGR);
    cvtColor(255 - references[indice], c, COLOR_GRAY2BGR);
    d = imageComparaison(im, det, references[indice], p.voisinage);

    Mat haut, bas, mosaique;
    hconcat(a, b, haut);
    hconcat(c, d, bas);
    vconcat(haut, bas, mosaique);
    resize(mosaique, mosaique, Size(), 0.75, 0.75);

    char texte[200];
    snprintf(texte, sizeof(texte), "%s - %s   P=%.3f  TFP=%.3f  TFN=%.3f  F=%.3f",
             noms[indice].c_str(), NOMS_DET[p.detecteur].c_str(), m.P, m.TFP, m.TFN, m.F);
    putText(mosaique, texte, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

    imshow("TP contours", mosaique);
}

void modeInteractif()
{
    namedWindow("TP contours");
    namedWindow("Parametres", WINDOW_NORMAL);
    resizeWindow("Parametres", 500, 350);

    createTrackbar("Image", "Parametres", &indice, images.size() - 1, afficher);
    createTrackbar("Detecteur", "Parametres", &p.detecteur, 2, afficher);
    createTrackbar("Flou", "Parametres", &p.flou, 12, afficher);
    createTrackbar("Seuil S/L", "Parametres", &p.seuil, 255, afficher);
    createTrackbar("Canny bas", "Parametres", &p.seuilBas, 255, afficher);
    createTrackbar("Canny haut", "Parametres", &p.seuilHaut, 255, afficher);
    createTrackbar("Noyau", "Parametres", &p.noyau, 2, afficher);
    createTrackbar("Voisinage", "Parametres", &p.voisinage, 3, afficher);

    afficher(0, 0);

    cout << "l : traiter tout le lot   s : sauvegarder   q : quitter" << endl;
    while (true) {
        int touche = waitKey(0);
        if (touche == 'q' || touche == 27 || touche == -1) break;

        if (touche == 'l') {
            Mesures moy = moyenne(traiterLot(p));
            cout << NOMS_DET[p.detecteur] << " sur " << images.size() << " images : P=" << moy.P
                 << " TFP=" << moy.TFP << " TFN=" << moy.TFN << " F=" << moy.F << endl;
        }
        if (touche == 's') {
            string nom = "resultats/" + noms[indice] + "_" + NOMS_DET[p.detecteur];
            Mat det = detecter(images[indice], p);
            imwrite(nom + "_contours.png", 255 - det);
            imwrite(nom + "_comparaison.png", imageComparaison(images[indice], det, references[indice], p.voisinage));
            cout << "Sauvegarde : " << nom << endl;
        }
    }
}

void balayage()
{
    filesystem::create_directories("resultats");
    ofstream f("resultats/balayage.csv");
    f << "detecteur;flou;seuil;seuil_bas;seuil_haut;P;TFP;TFN;precision;rappel;F\n";

    for (int det = 0; det < 3; det++) {
        Params meilleur;
        double meilleurP = -1;

        for (int flou = 0; flou <= 12; flou += 2) {
            vector<pair<int, int>> seuils;
            if (det == 2) {
                for (int h = 10; h <= 200; h += 10) {
                    seuils.push_back({h / 3, h});
                    seuils.push_back({h / 2, h});
                }
            } else {
                for (int s = 2; s <= 150; s += (s < 20 ? 2 : 5))
                    seuils.push_back({s, s});
            }

            for (auto& s : seuils) {
                Params q;
                q.detecteur = det;
                q.flou = flou;
                q.seuil = s.second;
                q.seuilBas = s.first;
                q.seuilHaut = s.second;

                Mesures m = moyenne(traiterLot(q));
                f << NOMS_DET[det] << ";" << flou << ";" << q.seuil << ";" << q.seuilBas << ";" << q.seuilHaut
                  << ";" << m.P << ";" << m.TFP << ";" << m.TFN << ";" << m.precision << ";" << m.rappel << ";" << m.F << "\n";

                if (m.P > meilleurP) {
                    meilleurP = m.P;
                    meilleur = q;
                }
            }
        }

        cout << NOMS_DET[det] << " : meilleur P=" << meilleurP << " (flou=" << meilleur.flou;
        if (det == 2) cout << ", bas=" << meilleur.seuilBas << ", haut=" << meilleur.seuilHaut << ")" << endl;
        else cout << ", seuil=" << meilleur.seuil << ")" << endl;
    }
}

int main(int argc, char** argv)
{
    chargerImages();
    if (images.empty()) return 0;

    string mode = argc > 1 ? argv[1] : "interactif";

    if (mode == "balayage") {
        balayage();
    } else if (mode == "lot") {
        if (argc < 5) {
            cout << "Usage : ./tp_contours lot <sobel|laplace|canny> <flou> <seuil> [seuil_haut] [voisinage]" << endl;
            return 0;
        }
        string d = argv[2];
        Params q;
        q.detecteur = d == "sobel" ? 0 : (d == "laplace" ? 1 : 2);
        q.flou = atoi(argv[3]);
        q.seuil = q.seuilBas = atoi(argv[4]);
        q.seuilHaut = argc > 5 ? atoi(argv[5]) : q.seuil;
        q.voisinage = argc > 6 ? atoi(argv[6]) : 1;

        string sortie = "resultats/" + d + "/";
        vector<Mesures> res = traiterLot(q, "resultats/lot_" + d + ".csv", sortie);

        cout << fixed << setprecision(3);
        for (size_t i = 0; i < res.size(); i++)
            cout << setw(14) << left << noms[i] << " P=" << res[i].P << " TFP=" << res[i].TFP << " TFN=" << res[i].TFN << " F=" << res[i].F << endl;
        Mesures moy = moyenne(res);
        cout << "MOYENNE        P=" << moy.P << " TFP=" << moy.TFP << " TFN=" << moy.TFN << endl;
        cout << "               precision=" << moy.precision << " rappel=" << moy.rappel << " F=" << moy.F << endl;
    } else {
        modeInteractif();
    }

    return 0;
}
