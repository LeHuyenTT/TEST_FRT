#include <iostream>
#include <opencv2/opencv.hpp>
#include "TMtCNN.h"
#include "TArcface.h"
#include "TRetina.h"
#include "TWarp.h"
#include "TLive.h"
#include "TBlur.h"
#include <fstream>
//----------------------------------------------------------------------------------------
//
// Created by markson zhang
//
// Edited by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
// Modified by DuongTT for testing 2023/07/20
//
//----------------------------------------------------------------------------------------

//! Build defines => comment them to turn a function off

// #define RETINA                   // comment if you want to use MtCNN landmark detection instead
#define RECOGNIZE_FACE
// #define TEST_LIVING              // no test fake or real
// #define BLUR_FILTER_STRANGER
// some diagnostics
#define SHOW_LEGEND
#define SHOW_LANDMARKS
//----------------------------------------------------------------------------------------
// Adjustable Parameters
//----------------------------------------------------------------------------------------
const int MaxItemsDatabase = 2000;
const int MinHeightFace = 90;
const float MinFaceThreshold = 0.50;
const float FaceLiving = 0.93;
const double MaxBlur = -25.0; // more positive = sharper image
const double MaxAngle = 10.0;
//----------------------------------------------------------------------------------------
// Some globals
//----------------------------------------------------------------------------------------
const int RetinaWidth = 320;
const int RetinaHeight = 240;
float ScaleX, ScaleY;
vector<std::string> NameFaces;
//----------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
//----------------------------------------------------------------------------------------
//  Computing the cosine distance between input feature and ground truth feature
//----------------------------------------------------------------------------------------
inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}
//----------------------------------------------------------------------------------------
// painting
//----------------------------------------------------------------------------------------
void DrawObjects(cv::Mat &frame, vector<FaceObject> &Faces)
{
    for (size_t i = 0; i < Faces.size(); i++)
    {
        FaceObject &obj = Faces[i];

        //----- rectangle around the face -------
        obj.rect.x *= ScaleX;
        obj.rect.y *= ScaleY;
        obj.rect.width *= ScaleX;
        obj.rect.height *= ScaleY;
        cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0));
//---------------------------------------

//----- diagnostic ----------------------
#ifdef SHOW_LANDMARKS
        for (int u = 0; u < 5; u++)
        {
            obj.landmark[u].x *= ScaleX;
            obj.landmark[u].y *= ScaleY;
        }

        cv::circle(frame, obj.landmark[0], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[1], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[2], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[4], 2, cv::Scalar(0, 255, 255), -1);
#endif // SHOW_LANDMARKS
//---------------------------------------
#ifdef SHOW_LEGEND
        cv::putText(frame, cv::format("Angle : %0.1f", obj.Angle), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));
        cv::putText(frame, cv::format("Face prob : %0.4f", obj.FaceProb), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));
        cv::putText(frame, cv::format("Name prob : %0.4f", obj.NameProb), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));
#ifdef TEST_LIVING
        if (obj.Color == 2)
        {
            // face is too tiny
            cv::putText(frame, cv::format("Live prob : ??"), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));
        }
        else
        {
            // face is ok
            cv::putText(frame, cv::format("Live prob : %0.4f", obj.LiveProb), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));
        }
#endif // TEST_LIVING
#endif // SHOW_LEGEND
        std::string Str;
        cv::Scalar color;
        int baseLine = 0;
        switch (obj.Color)
        {
        case 0:
            color = cv::Scalar(255, 255, 255);
            break; // default white -> face ok
        case 1:
            color = cv::Scalar(80, 255, 255);
            break; // yellow ->stranger
        case 2:
            color = cv::Scalar(255, 237, 178);
            break; // blue -> too tiny
        case 3:
            color = cv::Scalar(127, 127, 255);
            break; // red -> fake
        default:
            color = cv::Scalar(255, 255, 255);
        }
        switch (obj.NameIndex)
        {
        case -1:
            Str = "Stranger";
            break;
        case -2:
            Str = "too tiny";
            break;
        case -3:
            Str = "Fake !";
            break;
        default:
            Str = NameFaces[obj.NameIndex];
        }
        cv::Size label_size = cv::getTextSize(Str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > frame.cols)
            x = frame.cols - label_size.width;
        cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), color, -1);
        cv::putText(frame, Str, cv::Point(x, y + label_size.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0));
    }
}
//----------------------------------------------------------------------------------------
// main
//----------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    //=================================   START CODE INIT VARIABLE   =================================
    float f;
    float FPS[16];
    int n, numFrame, Fcnt = 0;
    size_t i;
    cv::Mat frame;
    cv::Mat result_cnn;
    cv::Mat faces;
    std::vector<FaceObject> Faces;
    vector<cv::Mat> fc1;
    string pattern_jpg = "../database/*.jpg";
    std::string NewItemName;
    size_t FaceCnt;
    TLive Live;
    TWarp Warp;
    TMtCNN MtCNN;
    TArcFace ArcFace;
    TRetina Rtn(RetinaWidth, RetinaHeight, true);
    TBlur Blur;
    chrono::steady_clock::time_point Tbegin, Tend;
    std::ofstream logFile("log.txt");
    if(!logFile.is_open()) {
        cerr<<" Error: Unable to open log file"<< endl;
        return 1;
    }
    string logTemp;
    Live.LoadModel();
    for (i = 0; i < 16; i++)
        FPS[i] = 0.0;
    //=================================    END CODE INIT VARIABLE   =================================

    //================================= START CODE ADD FACE TO DATABASE =================================
    if (argc > 1)
    {
        numFrame++;
        const char *imagepath = argv[1];
        cv::Mat frame = cv::imread(imagepath, 1);
        if (frame.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        ScaleX = ((float)frame.cols) / RetinaWidth;
        ScaleY = ((float)frame.rows) / RetinaHeight;
        // copy/resize image to result_cnn as input tensor
        cv::resize(frame, result_cnn, Size(RetinaWidth, RetinaHeight), INTER_LINEAR);
        // get the face
        MtCNN.detect(result_cnn, Faces);
        // only one face per picture
        if (Faces.size() == 1)
        {
            if (Faces[0].FaceProb > MinFaceThreshold)
            {
                // get centre aligned image
                cv::Mat aligned = Warp.Process(result_cnn, Faces[0]);
                std::string Str = imagepath;
                n = Str.rfind('/');
                Str = Str.erase(0, n + 1);
                Str = Str.erase(Str.length() - 4, Str.length() - 1); // remove .jpg
                imwrite("../database/" + Str + ".jpg", aligned);
                cout << "Stored to database : " << Str << endl;
            }
        }
        return 0;
    }
    //================================= END CODE ADD FACE TO DATABASE =================================

    //================================ START CODE LOAD DATABASE =================================
    cv::glob(pattern_jpg, NameFaces);
    FaceCnt = NameFaces.size();
    if (FaceCnt == 0)
    {
        cout << "No image files[jpg] in database" << endl;
    }
    else
    {
        cout << "Found " << FaceCnt << " pictures in database." << endl;
        for (i = 0; i < FaceCnt; i++)
        {
            // convert to landmark vector and store into fc
            faces = cv::imread(NameFaces[i]);
            fc1.push_back(ArcFace.GetFeature(faces));
            // get a proper name
            std::string &Str = NameFaces[i];
            n = Str.rfind('/');
            Str = Str.erase(0, n + 1);
            n = Str.find('#');
            if (n > 0)
                Str = Str.erase(n, Str.length() - 1); // remove # some numbers.jpg
            else
                Str = Str.erase(Str.length() - 4, Str.length() - 1); // remove .jpg
            if (FaceCnt > 1)
                printf("\rloading: %.2lf%% ", (i * 100.0) / (FaceCnt - 1));
        }
        cout << "" << endl;
        cout << "Loaded " << FaceCnt << " faces in total" << endl;
    }
    //================================ END CODE LOAD DATABASE =================================

    //================================ START CODE INIT CAMERA =================================
    cv::VideoCapture cap(0); // open camera 0
    if (!cap.isOpened())
    {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }
    cout << "Start grabbing, press ESC on TLive window to terminate" << endl;
    //================================ END CODE INIT CAMERA =================================

    //================================ START CODE INIT FACE RECOGNITION =================================
    while (1)
    {
        cap >> frame;
        if (frame.empty())
        {
            cerr << "Error: Unable to grab from the camera" << endl;
            break;
        }
        ScaleX = ((float)frame.cols) / RetinaWidth;
        ScaleY = ((float)frame.rows) / RetinaHeight;
        cv::resize(frame, result_cnn, Size(RetinaWidth, RetinaHeight), INTER_LINEAR);
        Tbegin = chrono::steady_clock::now();
        MtCNN.detect(result_cnn, Faces); //! Detect faces in picture by MTCNN
        // reset indicators
        for (i = 0; i < Faces.size(); i++)
        {
            Faces[i].NameIndex = -2; //-2 -> too tiny (may be negative to signal the drawing)
            Faces[i].Color = 2;
            Faces[i].NameProb = 0.0;
            Faces[i].LiveProb = 0.0;
        }
        // run through the faces only when you got one face.
        // more faces (if large enough) are not a problem
        // in this app with an input image of 324x240, they become too tiny
        if (Faces.size() == 1)
        {
            // looks stupid, running through a loop of size 1
            // however, for your convenience using [i]
            for (i = 0; i < Faces.size(); i++)
            {
                if (Faces[i].FaceProb > MinFaceThreshold)
                {
                    // get centre aligned image
                    cv::Mat aligned = Warp.Process(result_cnn, Faces[i]);
                    Faces[i].Angle = Warp.Angle;
                    // features of camera image
                    cv::Mat fc2 = ArcFace.GetFeature(aligned);
                    // reset indicators
                    Faces[i].NameIndex = -1; // a stranger
                    Faces[i].Color = 1;
                    // the similarity score
                    if (FaceCnt > 0)
                    {
                        vector<double> score_;
                        for (size_t c = 0; c < FaceCnt; c++)
                            score_.push_back(CosineDistance(fc1[c], fc2));
                        int Pmax = max_element(score_.begin(), score_.end()) - score_.begin();
                        Faces[i].NameIndex = Pmax;
                        Faces[i].NameProb = score_[Pmax];
                        score_.clear();
                        if (Faces[i].NameProb >= MinFaceThreshold)
                        {
                            // recognize a face
                            if (Faces[i].rect.height < MinHeightFace)
                            {
                                Faces[i].Color = 2; // found face in database, but too tiny
                            }
                            else
                            {
                                Faces[i].Color = 0; // found face in database and of good size
#ifdef TEST_LIVING
                                // test fake face
                                float x1 = Faces[i].rect.x;
                                float y1 = Faces[i].rect.y;
                                float x2 = Faces[i].rect.width + x1;
                                float y2 = Faces[i].rect.height + y1;
                                struct LiveFaceBox LiveBox = {x1, y1, x2, y2};
                                Faces[i].LiveProb = Live.Detect(result_cnn, LiveBox);
                                if (Faces[i].LiveProb <= FaceLiving)
                                {
                                    Faces[i].Color = 3; // fake
                                    Faces[i].NameIndex = -3;
                                }
#endif // TEST_LIVING
                            }
                        }
                        else
                        {
                            Faces[i].NameIndex = -1; // a stranger
                            Faces[i].Color = 1;
                        }
                    }

                    // test if the face is recognized, or should it be added to database
                    if (Faces[i].NameIndex == -1)
                    {
                        if (Faces[i].rect.height < MinHeightFace)
                        {
                            // a stranger with a small face
                            Faces[i].Color = 2; // too tiny
                        }
                    }
                }
            }
        }
        Tend = chrono::steady_clock::now();
        DrawObjects(frame, Faces);
        // calculate frame rate
        logTemp = "Frame: "+numFrame;
        logFile << logTemp;
        f = chrono::duration_cast<chrono::milliseconds>(Tend - Tbegin).count();
        logTemp +="time: " + to_string(f) + " ms\n";
        logFile << logTemp;
        if (f > 0.0)
            FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
        for (f = 0.0, i = 0; i < 16; i++)
        {
            f += FPS[i];
        }
        cv::putText(frame, cv::format("FPS %0.2f", f / 16), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));
        // show output
        cv::imshow("Jetson Nano - 2014.5 MHz", frame);
        char esc = cv::waitKey(5);
        if (esc == 27)
            break;
    }
    //================================ END CODE INIT FACE RECOGNITION =================================
    logFile.close();
    cv::destroyAllWindows();

    return 0;
}
