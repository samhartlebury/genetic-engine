#include "geneticprogram.h"

#include <QDebug>
#include <QThread>
#include <qmath.h>

GeneticProgram::GeneticProgram(QObject *parent) :
    QObject(parent),
    maxDepth(100)
{
    for (int i = 0; i < 3; ++i) {
        GeneticTree *tree = new GeneticTree;
        tree->maxInitialDepth = 100;
        m_genome.append(tree);
    }
}

GeneticProgram::~GeneticProgram()
{
    qDeleteAll(m_genome);
}

bool GeneticProgram::setMatrix(cv::Mat matrix)
{
    if (matrix.channels() < 3)
        return false;

    split(matrix, m_matrix);

    for (int i = 0; i < 3; ++i)
        m_genome[i]->setMatrix(m_matrix[i].clone());

    return true;
}

void GeneticProgram::setMaxInitialDepth(uint depth)
{
    maxDepth = depth;
}

bool GeneticProgram::generateGenome()
{
    for (const auto& tree : m_genome) {
        QThread::msleep(1);
        tree->maxInitialDepth = maxDepth;
        tree->generateTree();
    }

    return true;
}

GeneticProgram* GeneticProgram::breedWithProgram(GeneticProgram  * const program)
{
    GeneticProgram *child = new GeneticProgram;

    for (int i = 0; i < 3; ++i) {
        child->m_matrix[i] = m_matrix[i].clone();
        GeneticTree *baby = m_genome[i]->breedWithTree(program->m_genome[i]);
        if (!(qrand() % 5)) {// Temporary mutation rate (20%)
                baby->mutateRandomChild(baby);

        }

        *(child->m_genome[i]) = *(baby);
        delete baby;
    }


    return child;
}

cv::Mat GeneticProgram::evaluate()
{
    cv::Mat bgr[3];

    Q_ASSERT(m_genome[0] && m_genome[1] && m_genome[2]);

    for (int i = 0; i < 3; ++i) {
        cv::Mat debugger = m_genome[i]->evaluateTree();
        bgr[i] = debugger;
    }

    cv::Mat output;
    Q_ASSERT(bgr[0].cols);
    cv::merge(bgr, 3, output);

    return output;
}

qreal GeneticProgram::temperature(cv::Mat input)
{

    qreal c = 0.014404347826; // plancks constant * speed of light / boltzmann constant
    cv::Scalar rgb = cv::mean(input);

    for (int i = 0; i < 3; ++i) {
        if (rgb.val[i] == 0)
            rgb.val[i] = 0.000000000001;
    }

    // camera frequency responses /
    qreal lr = 0.000000580;
    qreal lg = 0.000000540;
    qreal lb = 0.000000450;
    // ------------------------- /

    qreal qrg = 1.0768; // qr/qg
    qreal qbg = 1.012;  // qb/qg

    qreal rt = 1 / ((1 / c) * ((lr * lg) / (lr - lg)) * qLn(((qPow(lg,-5))/(qPow(lr,-5))*(1/qrg)*(rgb.val[2]/rgb.val[1]))));
    qreal bt = 1 / ((1 / c) * ((lb * lg) / (lb - lg)) * qLn(((qPow(lg,-5))/(qPow(lb,-5))*(1/qbg)*(rgb.val[0]/rgb.val[1]))));

    qreal temperature = (rt+bt)/2;
    return temperature;

}

GeneticProgram &GeneticProgram::operator=(const GeneticProgram &source)
{
    // Check for self-assignment
    if (this == &source)
        return *this;

    // Deep copies
    for (int i = 0; i < 3; ++i) {
        m_matrix[i] = source.m_matrix[i].clone();
        *(m_genome[i]) = *(source.m_genome[i]);
    }


    // m_genome = source.m_genome;

    return *this;

}
