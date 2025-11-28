#include "modelmanager.h"
#include "modelwidget1.h"
#include "modelwidget2.h"
#include "modelwidget3.h"
#include "PressureDerivativeCalculator.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDebug>
#include <cmath>
#include <QtMath>
#include <algorithm>
#include <functional>

// ==================================================================================
//  ModelManager 构造与初始化
// ==================================================================================

ModelManager::ModelManager(QWidget* parent)
    : QObject(parent), m_mainWidget(nullptr), m_modelTypeCombo(nullptr), m_modelStack(nullptr)
    , m_modelWidget1(nullptr), m_modelWidget2(nullptr), m_modelWidget3(nullptr)
    , m_currentModelType(InfiniteConductive)
    , m_highPrecision(true) // 默认高精度
{
}

ModelManager::~ModelManager() {}

void ModelManager::initializeModels(QWidget* parentWidget)
{
    if (!parentWidget) return;
    createMainWidget();
    setupModelSelection();

    m_modelStack = new QStackedWidget(m_mainWidget);
    m_modelWidget1 = new ModelWidget1(m_modelStack);
    m_modelWidget2 = new ModelWidget2(m_modelStack);
    m_modelWidget3 = new ModelWidget3(m_modelStack);

    m_modelStack->addWidget(m_modelWidget1);
    m_modelStack->addWidget(m_modelWidget2);
    m_modelStack->addWidget(m_modelWidget3);

    m_mainWidget->layout()->addWidget(m_modelStack);
    connectModelSignals();
    switchToModel(InfiniteConductive);

    if (parentWidget->layout()) parentWidget->layout()->addWidget(m_mainWidget);
    else {
        QVBoxLayout* layout = new QVBoxLayout(parentWidget);
        layout->addWidget(m_mainWidget);
        parentWidget->setLayout(layout);
    }
}

void ModelManager::createMainWidget()
{
    m_mainWidget = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(m_mainWidget);
    mainLayout->setContentsMargins(10, 5, 10, 10);
    mainLayout->setSpacing(0);
    m_mainWidget->setLayout(mainLayout);
}

void ModelManager::setupModelSelection()
{
    if (!m_mainWidget) return;
    QGroupBox* selectionGroup = new QGroupBox("模型类型选择", m_mainWidget);
    selectionGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    QHBoxLayout* selectionLayout = new QHBoxLayout(selectionGroup);
    selectionLayout->setContentsMargins(9, 9, 9, 9);
    selectionLayout->setSpacing(6);

    QLabel* typeLabel = new QLabel("模型类型:", selectionGroup);
    typeLabel->setMinimumWidth(100);
    m_modelTypeCombo = new QComboBox(selectionGroup);
    m_modelTypeCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_modelTypeCombo->setMinimumWidth(200);

    m_modelTypeCombo->addItem(getModelTypeName(InfiniteConductive));
    m_modelTypeCombo->addItem(getModelTypeName(FiniteConductive));
    m_modelTypeCombo->addItem(getModelTypeName(SegmentedMultiCluster));

    m_modelTypeCombo->setStyleSheet("color: black;");
    typeLabel->setStyleSheet("color: black;");
    selectionGroup->setStyleSheet("QGroupBox { color: black; font-weight: bold; }");

    connect(m_modelTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ModelManager::onModelTypeSelectionChanged);
    selectionLayout->addWidget(typeLabel);
    selectionLayout->addWidget(m_modelTypeCombo);

    QVBoxLayout* mainLayout = qobject_cast<QVBoxLayout*>(m_mainWidget->layout());
    if (mainLayout) {
        mainLayout->addWidget(selectionGroup);
        mainLayout->setStretchFactor(selectionGroup, 0);
    }
}

void ModelManager::connectModelSignals()
{
    if (m_modelWidget1) connect(m_modelWidget1, &ModelWidget1::calculationCompleted, this, &ModelManager::onModel1CalculationCompleted);
    if (m_modelWidget2) connect(m_modelWidget2, &ModelWidget2::calculationCompleted, this, &ModelManager::onModel2CalculationCompleted);
    if (m_modelWidget3) connect(m_modelWidget3, &ModelWidget3::calculationCompleted, this, &ModelManager::onModel3CalculationCompleted);
}

void ModelManager::switchToModel(ModelType modelType)
{
    if (!m_modelStack) return;
    if (modelType == m_currentModelType) return;
    ModelType old = m_currentModelType;
    m_currentModelType = modelType;
    m_modelStack->setCurrentIndex((int)modelType);
    if (m_modelTypeCombo) m_modelTypeCombo->setCurrentIndex((int)modelType);
    emit modelSwitched(modelType, old);
}

void ModelManager::onModelTypeSelectionChanged(int index) { switchToModel((ModelType)index); }

QString ModelManager::getModelTypeName(ModelType type)
{
    switch (type) {
    case InfiniteConductive: return "无限导流双重孔隙介质页岩油藏渗流模型";
    case FiniteConductive: return "有限导流双重孔隙介质页岩油藏渗流模型";
    case SegmentedMultiCluster: return "分段多簇压裂水平井双重孔隙介质页岩油藏渗流模型";
    default: return "未知模型";
    }
}

QStringList ModelManager::getAvailableModelTypes()
{
    return { getModelTypeName(InfiniteConductive), getModelTypeName(FiniteConductive), getModelTypeName(SegmentedMultiCluster) };
}

void ModelManager::onModel1CalculationCompleted(const QString &t, const QMap<QString, double> &r) { emit calculationCompleted(t, r); }
void ModelManager::onModel2CalculationCompleted(const QString &t, const QMap<QString, double> &r) { emit calculationCompleted(t, r); }
void ModelManager::onModel3CalculationCompleted(const QString &t, const QMap<QString, double> &r) { emit calculationCompleted(t, r); }

// ==================================================================================
//  计算精度控制
// ==================================================================================
void ModelManager::setHighPrecision(bool high) {
    m_highPrecision = high;
}

// ==================================================================================
//  计算引擎核心实现
// ==================================================================================

QMap<QString, double> ModelManager::getDefaultParameters(ModelType type)
{
    QMap<QString, double> p;
    p.insert("k", 1.0);
    p.insert("S", 1.0);
    p.insert("cD", 1e-8);
    p.insert("N", 4.0);

    if (type == InfiniteConductive) {
        p.insert("omega", 0.05);
        p.insert("lambda", 0.1);
        p.insert("mf", 3.0);
        p.insert("nf", 5.0);
        p.insert("Xf", 40.0);
        p.insert("yy", 70.0);
        p.insert("y", 1000.0);
    }
    else if (type == FiniteConductive) {
        p.insert("omega", 0.0155);
        p.insert("lambda", 0.083);
        p.insert("mf", 3.0);
        p.insert("nf", 5.0);
        p.insert("Xf", 193.0);
        p.insert("yy", 295.0);
        p.insert("y", 2758.0);
        p.insert("CFD", 0.9);
        p.insert("kpd", 0.04);
        p.insert("S", 0.81);
        p.insert("cD", 8.08e-8);
    }
    else if (type == SegmentedMultiCluster) {
        p.insert("omega1", 0.05);
        p.insert("omega2", 0.05);
        p.insert("lambda1", 1e-1);
        p.insert("lambda2", 1e-1);
        p.insert("mf1", 2.0);
        p.insert("mf2", 2.0);
        p.insert("nf", 5.0);
        p.insert("Xf1", 40.0);
        p.insert("Xf2", 40.0);
        p.insert("yy1", 70.0);
        p.insert("yy2", 70.0);
        p.insert("y", 800.0);
        p.insert("CFD1", 0.4);
        p.insert("CFD2", 0.4);
        p.insert("kpd", 0.045);
    }
    return p;
}

QVector<double> ModelManager::generateLogTimeSteps(int count, double startExp, double endExp) {
    QVector<double> t;
    t.reserve(count);
    for (int i = 0; i < count; ++i) {
        double exponent = startExp + (endExp - startExp) * i / (count - 1);
        t.append(pow(10.0, exponent));
    }
    return t;
}

ModelCurveData ModelManager::calculateTheoreticalCurve(ModelType type, const QMap<QString, double>& params, const QVector<double>& providedTime)
{
    QVector<double> tPoints = providedTime;
    if (tPoints.isEmpty()) {
        tPoints = generateLogTimeSteps(100, -7.0, 7.0);
    }

    switch(type) {
    case InfiniteConductive:
        return calculateModel1(params, tPoints);
    case FiniteConductive:
        return calculateModel2(params, tPoints);
    case SegmentedMultiCluster:
        return calculateModel3(params, tPoints);
    default:
        return calculateModel1(params, tPoints);
    }
}

void ModelManager::calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                                       std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                                       QVector<double>& outPD, QVector<double>& outDeriv)
{
    int numPoints = tD.size();
    outPD.resize(numPoints);
    outDeriv.resize(numPoints);

    // 拟合模式下，强制使用较小的 N 进一步加速 (例如 N=4)
    // 高精度模式使用参数设定的 N
    int N_param = (int)params.value("N", 4);
    int N = m_highPrecision ? N_param : 4;
    if (N % 2 != 0) N = 4;

    double ln2 = log(2.0);

    for (int k = 0; k < numPoints; ++k) {
        double t = tD[k];
        if (t <= 1e-10) {
            outPD[k] = 0;
            continue;
        }

        double pd_val = 0.0;
        for (int m = 1; m <= N; ++m) {
            double s = m * ln2 / t;
            double L = laplaceFunc(s, params);
            pd_val += stefestCoefficient(m, N) * ln2 * L / t;
        }
        outPD[k] = pd_val;
    }

    double lSpacing = 0.1;
    if (numPoints > 2) {
        outDeriv = PressureDerivativeCalculator::calculateBourdetDerivative(tD, outPD, lSpacing);
    } else {
        outDeriv.fill(0.0);
    }
}

ModelCurveData ModelManager::calculateModel1(const QMap<QString, double>& params, const QVector<double>& tPoints)
{
    QVector<double> pd, dpd;
    auto func = std::bind(&ModelManager::flaplace1, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tPoints, params, func, pd, dpd);
    return std::make_tuple(tPoints, pd, dpd);
}

ModelCurveData ModelManager::calculateModel2(const QMap<QString, double>& params, const QVector<double>& tPoints)
{
    QVector<double> pd, dpd;
    auto func = std::bind(&ModelManager::flaplace2, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tPoints, params, func, pd, dpd);

    double kpd = params.value("kpd", 0.0);
    if (std::abs(kpd) > 1e-5) {
        for(int i=0; i<pd.size(); ++i) {
            double raw_p = pd[i];
            pd[i] = -1.0 / kpd * log(1.0 - kpd * raw_p);
            if ((1.0 - kpd * raw_p) > 1e-9)
                dpd[i] = dpd[i] / (1.0 - kpd * raw_p);
        }
    }
    return std::make_tuple(tPoints, pd, dpd);
}

ModelCurveData ModelManager::calculateModel3(const QMap<QString, double>& params, const QVector<double>& tPoints)
{
    return calculateModel1(params, tPoints);
}

// ==================================================================================
//  数学核心：高精度积分与辅助函数
// ==================================================================================

static const double GL15_X[] = { 0.0, 0.2011940939974345, 0.3941513470775634, 0.5709721726085388, 0.7244177313601701, 0.8482065834104272, 0.9372985251687639, 0.9879925180204854 };
static const double GL15_W[] = { 0.2025782419255613, 0.1984314853271116, 0.1861610000155622, 0.1662692058169939, 0.1395706779049514, 0.1071592204671719, 0.0703660474881081, 0.0307532419961173 };

double ModelManager::gauss15(std::function<double(double)> f, double a, double b) {
    double halfLen = 0.5 * (b - a);
    double center = 0.5 * (a + b);
    double sum = GL15_W[0] * f(center);
    for (int i = 1; i < 8; ++i) {
        double dx = halfLen * GL15_X[i];
        sum += GL15_W[i] * (f(center - dx) + f(center + dx));
    }
    return sum * halfLen;
}

double ModelManager::adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth) {
    double c = (a + b) / 2.0;
    double v1 = gauss15(f, a, b);
    double v2 = gauss15(f, a, c) + gauss15(f, c, b);

    if (depth >= maxDepth) return v2;
    if (std::abs(v1 - v2) < 1e-10 * std::abs(v2) + eps) return v2;
    return adaptiveGauss(f, a, c, eps/2, depth+1, maxDepth) + adaptiveGauss(f, c, b, eps/2, depth+1, maxDepth);
}

double ModelManager::integralBesselK0(double XDkv, double YDkv, double yDij, double fz, double a, double b) {
    // -----------------------------------------------------------
    // 极速优化核心：低精度模式下大幅简化积分
    // -----------------------------------------------------------
    if (!m_highPrecision) {
        // 低精度模式：只处理明显的奇异点，否则直接使用 Gauss-15
        double dist_y_sq = (YDkv - yDij) * (YDkv - yDij);
        if (dist_y_sq < 1e-16 && XDkv > a && XDkv < b) {
            // 奇异点在区间内，简单切分
            auto func = [=](double xwD) {
                double dist_x = XDkv - xwD;
                return besselK0(sqrt(dist_x * dist_x) * sqrt(fz));
            };
            return gauss15(func, a, XDkv) + gauss15(func, XDkv, b);
        }

        // 普通情况直接积分
        auto func = [=](double xwD) {
            double dist_x = XDkv - xwD;
            double arg = sqrt(dist_x * dist_x + dist_y_sq) * sqrt(fz);
            return besselK0(arg);
        };
        return gauss15(func, a, b);
    }

    // -----------------------------------------------------------
    // 高精度模式：完整自适应逻辑
    // -----------------------------------------------------------
    double sqrt_fz = sqrt(fz);
    double dist_y_sq = (YDkv - yDij) * (YDkv - yDij);

    auto func = [=](double xwD) -> double {
        double dist_x = XDkv - xwD;
        double arg = sqrt(dist_x * dist_x + dist_y_sq) * sqrt_fz;
        return besselK0(arg);
    };

    double segmentLen = b - a;
    double distToCenter = sqrt(pow(XDkv - (a+b)/2.0, 2) + dist_y_sq);

    if (distToCenter > 2.0 * segmentLen) {
        return gauss15(func, a, (a+b)/2.0) + gauss15(func, (a+b)/2.0, b);
    }
    if (dist_y_sq < 1e-16) {
        if (XDkv > a + 1e-9 && XDkv < b - 1e-9) {
            return adaptiveGauss(func, a, XDkv, 1e-7, 0, 12) + adaptiveGauss(func, XDkv, b, 1e-7, 0, 12);
        }
    }
    return adaptiveGauss(func, a, b, 1e-7, 0, 12);
}

double ModelManager::besselK0(double x) {
    if (x <= 1e-15) return 50.0;
#if __cplusplus >= 201703L
    return std::cyl_bessel_k(0, x);
#else
    if (x <= 2.0) {
        double y = x * x / 4.0;
        return (-log(x / 2.0) * std::cyl_bessel_i(0, x)) +
               (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))));
    } else {
        double y = 2.0 / x;
        return (exp(-x) / sqrt(x)) * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))));
    }
#endif
}

QVector<double> ModelManager::solveLinearSystem(QVector<QVector<double>> A, QVector<double> b) {
    int n = b.size();
    for(int i=0; i<n; ++i) A[i].append(b[i]);
    for (int k = 0; k < n - 1; ++k) {
        if (std::abs(A[k][k]) < 1e-12) continue;
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n + 1; ++j) A[i][j] -= factor * A[k][j];
        }
    }
    QVector<double> x(n);
    if (std::abs(A[n-1][n-1]) > 1e-12) x[n-1] = A[n-1][n] / A[n-1][n-1];
    for (int i = n - 2; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) sum += A[i][j] * x[j];
        if (std::abs(A[i][i]) > 1e-12) x[i] = (A[i][n] - sum) / A[i][i];
    }
    return x;
}

double ModelManager::stefestCoefficient(int i, int N) {
    double sum = 0.0;
    int k_start = (i + 1) / 2;
    int k_end = std::min(i, N / 2);
    for (int k = k_start; k <= k_end; ++k) {
        double num = pow(k, N / 2.0) * factorial(2 * k);
        double den = factorial(N / 2 - k) * factorial(k) * factorial(k - 1) * factorial(i - k) * factorial(2 * k - i);
        if(den != 0) sum += num / den;
    }
    return ((i + N / 2) % 2 == 0 ? 1.0 : -1.0) * sum;
}

double ModelManager::factorial(int n) {
    if (n <= 1) return 1.0;
    double res = 1.0;
    for (int i = 2; i <= n; ++i) res *= i;
    return res;
}

// ==================================================================================
//  Model 1 专用函数
// ==================================================================================
double ModelManager::flaplace1(double z, const QMap<QString, double>& p) {
    int mf = (int)p.value("mf", 3);
    int nf = (int)p.value("nf", 5);
    double omega = p.value("omega", 0.05);
    double lambda = p.value("lambda", 0.1);
    double Xf = p.value("Xf", 40.0);
    double yy = p.value("yy", 70.0);
    double y = p.value("y", 1000.0);

    if (mf <= 0 || nf <= 0) return 0.0;

    int sz = mf * 2 * nf;
    QVector<QVector<double>> E(sz + 1, QVector<double>(sz + 1, 0.0));
    QVector<double> F(sz + 1, 0.0);

    for(int i=1; i<=mf; ++i) {
        for(int j=1; j<=2*nf; ++j) {
            for(int k=1; k<=mf; ++k) {
                for(int v=1; v<=2*nf; ++v) {
                    E[(i-1)*2*nf+j-1][(k-1)*2*nf+v-1] = e_function(z,i,j,k,v,mf,nf,omega,lambda,Xf,yy,y);
                }
            }
            E[(i-1)*2*nf+j-1][sz] = -1.0;
            E[sz][(i-1)*2*nf+j-1] = f_function(j,nf,Xf,y);
        }
    }
    F[sz] = 1.0/z;

    QVector<double> res = solveLinearSystem(E, F);
    if (res.size() <= sz) return 0.0;

    return res[sz];
}

double ModelManager::flaplace2(double z, const QMap<QString, double>& p) {
    int mf = (int)p.value("mf", 3);
    int nf = (int)p.value("nf", 5);
    double omega = p.value("omega", 0.0155);
    double lambda = p.value("lambda", 0.083);
    double Xf = p.value("Xf", 193.0);
    double yy = p.value("yy", 295.0);
    double y = p.value("y", 2758.0);
    double CFD = p.value("CFD", 0.9);

    if(mf<=0 || nf<=0) return 0.0;
    int sz = mf * nf;
    double deltaL = Xf / (nf * y);

    QVector<QVector<double>> I(sz+mf+1, QVector<double>(sz+mf+1));
    QVector<double> F(sz+mf+1, 0);

    for(int i=1; i<=mf; ++i) {
        for(int j=1; j<=nf; ++j) {
            int r = (i-1)*nf + j - 1;
            for(int k=1; k<=mf; ++k) {
                for(int v=1; v<=nf; ++v) {
                    double val = integralBesselK0(
                        ((2*v-2*nf-1)/(2.0*nf)*Xf)/y, (yy+(y-2*yy)/(mf-1)*(k-1))/y, (yy+(y-2*yy)/(mf-1)*(i-1))/y,
                        z*(omega*z*(1-omega)+lambda)/(lambda+(1-omega)*z), ((j-nf-1)/(double)nf*Xf)/y, ((j-nf)/(double)nf*Xf)/y
                        );
                    if(i==k) val -= (j==v ? M_PI/(CFD*8)*deltaL*deltaL : M_PI/CFD*std::abs(j-v)*deltaL*deltaL);
                    I[r][(k-1)*nf + v - 1] = val;
                }
            }
            I[r][sz] = (2*M_PI/CFD) * (((j-nf)/(double)nf*Xf/y) - ((j-nf-1)/(double)nf*Xf/y));
            I[r][sz+mf] = -1.0;
        }
    }
    for(int i=1; i<=mf; ++i) {
        for(int j=(i-1)*nf; j<i*nf; ++j) I[sz+i-1][j] = deltaL;
        I[sz+i-1][sz+i-1] = -1.0;
    }
    for(int i=sz; i<sz+mf; ++i) I[sz+mf][i] = 1.0;

    F[sz+mf] = 1.0/z;
    QVector<double> res = solveLinearSystem(I, F);
    if(res.size() <= sz+mf) return 0.0;

    return res[sz+mf];
}

double ModelManager::e_function(double z, int i, int j, int k, int v, int mf, int nf, double omega, double lambda, double Xf, double yy, double y)
{
    double fz = (z * (omega * z * (1 - omega) + lambda)) / (lambda + (1 - omega) * z);
    double xij = (j - nf - 1) / (double)nf * Xf;
    double xDij = xij / y;
    double xij1 = (j - nf) / (double)nf * Xf;
    double xDij1 = xij1 / y;
    double Xkv = (2*v - 2*nf - 1) / (double)(2*nf) * Xf;
    double XDkv = Xkv / y;
    double yij = yy + (y - 2*yy) / (mf - 1) * (i - 1);
    double yDij = yij / y;
    double Ykv = yy + (y - 2*yy) / (mf - 1) * (k - 1);
    double YDkv = Ykv / y;
    return integralBesselK0(XDkv, YDkv, yDij, fz, xDij, xDij1);
}

double ModelManager::f_function(int j, int nf, double Xf, double y)
{
    double xij = (j - nf - 1) / (double)nf * Xf;
    double xDij = xij / y;
    double xij1 = (j - nf) / (double)nf * Xf;
    double xDij1 = xij1 / y;
    return xDij1 - xDij;
}
