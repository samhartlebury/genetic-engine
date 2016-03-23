#include "geneticengine.h"
#include "genetictree.h"
#include <opencv2/opencv.hpp>

#include <QDebug>
#include <QDateTime>
#include <QThread>

using namespace cv;

GeneticEngine::GeneticEngine(int argc, char *argv[]) :
    QApplication(argc, argv),
    population(500),
    breedingPoolSize(50),
    generations(10),
    initialDepth(20)
{
    Q_UNUSED(argc);
    Q_UNUSED(argv);
}

bool lowestError(GeneticEngine::GeneticData* a, GeneticEngine::GeneticData* b)
{
    return a->error < b->error;
}

void GeneticEngine::firstGeneration()
{

    qsrand(QDateTime::currentDateTime().toMSecsSinceEpoch());

    for (int i = 0; i < population; ++i) {


        processEvents();

        GeneticProgram *program = new GeneticProgram;
        program->setMatrix(input);
        program->setMaxInitialDepth(initialDepth);
        program->generateGenome();

        Mat output = program->evaluate().clone();
        Mat diff;

        absdiff(target, output, diff);

        Scalar bgr = mean(diff);
        qreal error = qMax(qMax(bgr[0], bgr[1]), bgr[2]);

        GeneticData *data = new GeneticData;
        data->error = error;
        data->output = output.clone();
        *(data->program) = *program;

        bestList.append(data);

        qDebug() << QString::number((double(i) / double(population)) * 100.00)
                 << "%" << bestList.at(0)->error
                 << program->m_genome.at(0)->depthOfTree();


        if (bestList.size() > breedingPoolSize) {
            std::sort(bestList.begin(), bestList.end(), lowestError);
            delete bestList.last();
            bestList.removeLast();
        }

        delete program;
    }
}

void GeneticEngine::nextGeneration()
{
    qDeleteAll(newBestList);
    newBestList.clear();
    newBestList = bestList; // Shallow copy data to new list
    bestList.clear(); // Reset for next generation


    for (int i = 0; i < population; ++i) {

        int thisElement = i % breedingPoolSize;
        int randomElement = (qrand()) % breedingPoolSize;

        while (randomElement == thisElement)
            randomElement = (qrand()) % breedingPoolSize;

        const auto program1 = newBestList[thisElement]->program;
        const auto program2 = newBestList[randomElement]->program;

        Q_ASSERT(program1 && program2);

        GeneticData *data = new GeneticData;
        data->program = program1->breedWithProgram(program2);
        Mat output = data->program->evaluate();
        Mat diff;
        absdiff(target, output, diff);
        Scalar bgr = mean(diff);
        qreal error = (bgr[0] + bgr[1] + bgr[2]) / 3;
        data->error = error;
        data->output = output;

        bestList.append(data);

        if (bestList.size() > breedingPoolSize) {
            std::sort(bestList.begin(), bestList.end(), lowestError);
            delete bestList.last();
            bestList.removeLast();
        }

        qDebug() << QString::number((double(i) / double(population)) * 100.00)
                 << "%" << bestList.at(0)->error
                 << bestList.at(0)->program->m_genome.at(0)->depthOfTree();
    }
}

void GeneticEngine::medianError()
{
    if (bestList.isEmpty())
        return;

    qDebug() << endl << "Median error:";

    if ((bestList.size() % 2) == 0) { // Even number
        qreal median = (bestList.at(bestList.size() / 2)->error / 255) * 100;
        qDebug() << QString::number(median);
    } else if (bestList.size() == 1) {
        qDebug() << QString::number((bestList.at(0)->error / 255) * 100);
    } else { // Odd number
        qreal median1 = (bestList.at((bestList.size() - 1) / 2)->error / 255) * 100;
        qreal median2 = (bestList.at((bestList.size() + 1) / 2)->error / 255) * 100;
        qreal median = (median1 + median2) / 2;
        qDebug() << QString::number(median);
    }
}

void GeneticEngine::start()
{
    Mat preInput = imread("/home/sam/Pictures/test3.png",CV_LOAD_IMAGE_COLOR);
    Mat preTarget = imread("/home/sam/Pictures/test2.png", CV_LOAD_IMAGE_COLOR);

    int divider = 4;

    resize(preInput, input, Size(preInput.cols / divider, preInput.rows / divider));
    resize(preTarget, target, Size(preTarget.cols / divider, preTarget.rows / divider));

    imshow("input", input);
    imshow("target", target);
    target.convertTo(target, CV_32F);

    ResultsLog logger("/home/sam/results.txt");

    if (generations > 0) {
        firstGeneration();
        logger.writeCurrentData(1, bestList);
    }
    for (int i = 0; i < (generations - 1); ++i) {
        nextGeneration();
        logger.writeCurrentData(i + 2, bestList);
    }

    qDebug() << endl << "Best error"
             << endl << (bestList.at(0)->error / 255) * 100;

    medianError();

    Mat best = bestList.at(0)->output;
    best.convertTo(best, CV_8U);

    imshow("best", best);
    imwrite( "/home/sam/Pictures/bestTest.png", best);
    processEvents();

    // Testing reconstruction

    Mat newInput =  imread("/home/sam/Pictures/objects.png",CV_LOAD_IMAGE_COLOR);
    resize(newInput, newInput, Size(preTarget.cols / divider, preTarget.rows / divider));

    imshow("input with objects", newInput);

    auto bestProgram = bestList.at(0)->program;
    bestProgram->setMatrix(newInput);
    Mat newBest = bestProgram->evaluate();
    newBest.convertTo(newBest, CV_8U);

    imshow("output with objects", newBest);
    imwrite( "/home/sam/Pictures/bestTestObjects.png", newBest);

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
    // Deep copys
    output = source.output.clone();
    *program = *(source.program);
    return *this;

}

GeneticEngine::GeneticData::~GeneticData()
{
    delete program;
}

GeneticEngine::ResultsLog::ResultsLog(const QString &filePath) :
    file(filePath)
{
    if (!file.open(QIODevice::WriteOnly))
        Q_ASSERT(false);

    out.setDevice(&file);   // we will serialize the data into the file
}

void GeneticEngine::ResultsLog::writeCurrentData(int generation, const QList<GeneticData*> &bestList)
{
    out << "Generation: " << generation << endl;
    out << "Best error: " << (bestList.at(0)->error / 255) * 100 << endl;
    out << "Median error: ";

    if ((bestList.size() % 2) == 0) { // Even number
        qreal median = (bestList.at(bestList.size() / 2)->error / 255) * 100;
        out << QString::number(median);
    } else if (bestList.size() == 1) {
        out << QString::number((bestList.at(0)->error / 255) * 100);
    } else { // Odd number
        qreal median1 = (bestList.at((bestList.size() - 1) / 2)->error / 255) * 100;
        qreal median2 = (bestList.at((bestList.size() + 1) / 2)->error / 255) * 100;
        qreal median = (median1 + median2) / 2;
        out << QString::number(median);
    }

    out << endl << endl;
}
