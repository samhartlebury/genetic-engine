#ifndef GENETICENGINE_H
#define GENETICENGINE_H

#include <QApplication>

#include "geneticprogram.h"

class GeneticEngine : public QApplication
{
    Q_OBJECT

public:
    GeneticEngine(int argc, char *argv[]);

    struct GeneticData {
        GeneticData();
        GeneticProgram *program;
        cv::Mat output;
        qreal error;

        GeneticData& operator=(const GeneticData &source);
    };

    int population;
    int breedingPoolSize;
    int generations;

    QList<GeneticData> bestList;

public slots:
    void start();
};

#endif // GENETICENGINE_H
