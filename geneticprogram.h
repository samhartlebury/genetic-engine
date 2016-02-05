#ifndef GENETICPROGRAM_H
#define GENETICPROGRAM_H

#include <QObject>
#include "genetictree.h"

class GeneticProgram : public QObject
{
    Q_OBJECT
public:
    explicit GeneticProgram(QObject *parent = 0);
    bool setMatrix(cv::Mat matrix);
    bool generateGenome();
    cv::Mat evaluate();
    qreal temperature(cv::Mat input);

    GeneticProgram& operator=(const GeneticProgram &source);

    cv::Mat m_matrix[3];
    QList<GeneticTree*> m_genome;
    GeneticProgram *breedWithProgram(GeneticProgram * const program);
};

#endif // GENETICPROGRAM_H
