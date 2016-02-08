#ifndef GENETICENGINE_H
#define GENETICENGINE_H

#include <QApplication>

#include "geneticprogram.h"

class GeneticEngine : public QApplication
{
    Q_OBJECT

    void firstGeneration();
    void nextGeneration();
    void medianError();
    
public:
    GeneticEngine(int argc, char *argv[]);

    cv::Mat input;
    cv::Mat target;

    struct GeneticData {
        GeneticData();
        GeneticProgram *program;
        cv::Mat output;
        qreal error;

        GeneticData& operator=(const GeneticData &source);
        ~GeneticData();
    };

    int population;
    int breedingPoolSize;
    int generations;
    int initialDepth;

    QList<GeneticData*> bestList;
    QList<GeneticData*> newBestList;

public slots:
    void start();
};

#endif // GENETICENGINE_H
