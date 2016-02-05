#include "geneticengine.h"
#include "genetictree.h"
#include <opencv2/opencv.hpp>

#include <QDebug>

using namespace cv;

GeneticEngine::GeneticEngine(int argc, char *argv[]) :
    QApplication(argc, argv),
    population(100),
    breedingPoolSize(10),
    generations(2)
{
    Q_UNUSED(argc);
    Q_UNUSED(argv);
}

bool lowestError(GeneticEngine::GeneticData a, GeneticEngine::GeneticData b)
{
    return a.error < b.error;
}

void GeneticEngine::start()
{
    Mat input = imread("/home/sam/opencv2.png",CV_LOAD_IMAGE_COLOR);
    Mat target = imread("/home/sam/opencv.png", CV_LOAD_IMAGE_COLOR);
    target.convertTo(target, CV_32F);

    for (int i = 0; i < population; ++i) {
        GeneticProgram *program = new GeneticProgram;
        program->setMatrix(input);
        program->generateGenome();

        Mat output = program->evaluate();
        Mat diff;

        absdiff(target, output, diff);

        Scalar bgr = mean(diff);
        qreal error = (bgr[0] + bgr[1] + bgr[2]) / 3;

        GeneticData *data = new GeneticData;
        data->error = error;
        data->output = output.clone();
        *(data->program) = *program;

        qDebug() << error;

        bestList.append(*data);

        if (bestList.size() > breedingPoolSize) {
            std::sort(bestList.begin(), bestList.end(), lowestError);
            bestList.removeLast();
        }

        delete data;
        delete program;
    }

    qDebug() << endl << "Best errors:";

    for (auto data : bestList)
        qDebug() << data.error;

    auto baby = bestList.at(0).program->breedWithProgram(bestList[1].program);

    imshow("input", input);
    imshow("target", target);
    imshow("best", baby->evaluate());
}

GeneticEngine::GeneticData::GeneticData() :
    program(new GeneticProgram)
{
}

GeneticEngine::GeneticData &GeneticEngine::GeneticData::operator=(const GeneticEngine::GeneticData &source)
{
    // Check for self-assignment
    if (this == &source)
        return *this;

    // Shallow copy source non-pointers
    error = source.error;
    output = source.output.clone();
    program = source.program;
    return *this;

}
