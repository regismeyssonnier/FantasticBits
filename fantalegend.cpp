#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx")  //Enable AVX
#include <x86intrin.h> //AVX/SSE Extensions

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <set>
#include <random>
#include <chrono>
#include <deque>
#include <queue>
#include <assert.h>
#include <memory>
using namespace std::chrono;
using namespace std;

#define PI 3.1415926535897932384626433832795

#define ll long long int

struct Vector2{
    float x;
    float y;
    Vector2(){};
    Vector2(float xx, float yy):x(xx), y(yy){};

    Vector2 operator-(const Vector2& other) const { return {x - other.x, y - other.y}; }
    Vector2 operator+(const Vector2& other) const { return {x + other.x, y + other.y}; }
    Vector2 operator*(double scalar) const { return {x * scalar, y * scalar}; }

    double length() const { return sqrt(x * x + y * y); }
    double lengthSquared() const { return x * x + y * y; }
    
};

inline double dist(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

inline double dist2(double x1, double y1, double x2, double y2) {
    return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
}

inline double dist(Vector2 p, double x2, double y2) {
    return dist(p.x, p.y, x2, y2);
}

inline double dist2(Vector2 p, double x2, double y2) {
    return dist2(p.x, p.y, x2, y2);
}

inline double dist(Vector2 u1, Vector2 u2) {
    return dist(u1.x, u1.y, u2.x, u2.y);
}

inline double dist2(Vector2 u1, Vector2 u2) {
    return dist2(u1.x, u1.y, u2.x, u2.y);
}



inline float distance(Vector2 v1,Vector2 v2){
    return sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y));
}

inline float norme1(Vector2 v){
    return sqrt(v.x*v.x + v.y*v.y);
}

inline float dot(const Vector2 &a, const Vector2 &b) {
  return a.x*b.x+a.y*b.y;
}
inline float cross(const Vector2 &vec, const Vector2 &axe) {
	//projeté de vec sur la direction orthogonale à axe, à +90°
  return vec.y*axe.x-vec.x*axe.y;
}

inline float crossproduct(Vector2 a, Vector2 b){
    return a.x * b.y - a.y * a.x;
}

constexpr double WIDTH = 16000.0;
constexpr double HEIGHT = 7500.0;

constexpr int OBLIVIATE = 0;
constexpr int PETRIFICUS = 1;
constexpr int ACCIO = 2;
constexpr int FLIPENDO = 3;
constexpr int SPELL_DURATIONS[] = {3, 1, 6, 3};
constexpr int SPELL_COSTS[] = {5, 10, 20, 20};

constexpr double E = 0.00001;
constexpr int VERTICAL = 1;
constexpr int HORIZONTAL = 2;
constexpr double INF = 16000*16000 + 7500*7500;
constexpr int WIZARD = 1;
constexpr int SNAFFLE = 2;
constexpr int BLUDGER = 3;
constexpr int POLE = 4;
constexpr double TO_RAD = M_PI / 180.0;

double cosAngles[360];
double sinAngles[360];

static unsigned int g_seed;
inline void fast_srand(int seed) {
  //Seed the generator
  g_seed = seed;
}
inline int fastrand() {
  //fastrand routine returns one integer, similar output value range as C lib.
  g_seed = (214013*g_seed+2531011);
  return (g_seed>>16)&0x7FFF;
}
inline int fastRandInt(int maxSize) {
  return fastrand() % maxSize;
}
inline int fastRandInt(int a, int b) {
  return(a + fastRandInt(b - a));
}
inline double fastRandDouble() {
  return static_cast<double>(fastrand()) / 0x7FFF;
}
inline double fastRandDouble(double a, double b) {
  return a + (static_cast<double>(fastrand()) / 0x7FFF)*(b-a);
}



struct Compare
{
    bool operator()(const pair<double, pair<Vector2, vector<Vector2>>>& a, const pair<double, pair<Vector2, vector<Vector2>>>& b)
    {
       return a.first > b.first;
    }
};

Vector2 VEND1, VEND2;


class Sim{
public:
    int id;
    //int x;
    //int y;
    int state;
    int thrust;
    double angle, anglet;
    Vector2 pos;
    Vector2 speed;
    int idteam ;
    int xt;
    int yt;
    int thrustt;
    double radius;
    int idsnaffle=-1;
    int idcapt = -1;
    int sort;
    int id_sort;
    bool coord = false;
    int id_ball = -1;
    bool have_in_main = false;
    int targetsc = -1;
    int use_all = 0;
    int is_def = false;

    int type;

    bool dead = false;
    double r;
    double m;
    double f;
    
    //double vx;
    //double vy;
    //Vector2 speed;

    double sx;
    double sy;
    double svx;
    double svy;

    //Sim carrier;
    //Sim snaffle;

    int grab = 0;

//#ifndef PROD
    double lx;
    double ly;
    double lvx;
    double lvy;
    //Sim lcarrier;
    //Sim lsnaffle;
    bool ldead;

    void store() {
        lx = pos.x;
        ly = pos.y;
        lvx = speed.x;
        lvy = speed.y;
        //lcarrier = carrier;
        //lsnaffle = snaffle;
        ldead = dead;
    }

    void compare();
//#endif

    void update(int id, int x, int y, int vx, int vy, int state) {
        this->id = id;
        this->pos.x = x;
        this->pos.y = y;
        this->speed.x = vx;
        this->speed.y = vy;
        this->state = state;
    }

    double speedTo(Vector2 p) {
        double d = 1.0 / dist(this->pos, pos.x, pos.y);

        // vitesse dans la direction du checkpoint - (vitesse orthogonale)^2/dist au cheeckpoint
        double dx = (p.x - this->pos.x) * d;
        double dy = (p.y - this->pos.y) * d;
        double nspeed = speed.x * dx + speed.y * dy;
        double ospeed = dy * speed.x - dx * speed.y;

        return nspeed - (5 * ospeed * ospeed * d);
    }

    double speeds() {
        return sqrt(speed.x * speed.x + speed.y * speed.y);
    }

    inline void thrusts(double thrust, Sim p, double distance, double mass) {
        double coef = (thrust / m) / distance;
        speed.x += (p.pos.x - pos.x) * coef;
        speed.y += (p.pos.y - pos.y) * coef;
    }

    inline void thrusts(double thrust, double x, double y, double distance, double mass) {
        double coef = (thrust / mass) / distance;
        speed.x += (x - this->pos.x) * coef;
        speed.y += (y - this->pos.y) * coef;
    }

    virtual void move(double t) {
        pos.x += speed.x * t;
        pos.y += speed.y * t;
    }

    virtual /*Collision**/ double collision(double from) {
        double tx = 2.0;
        double ty = tx;

        if (pos.x + speed.x < r) {
            tx = (r - pos.x) / speed.x;
        }
        else if (pos.x + speed.x > WIDTH - r) {
            tx = (WIDTH - r - pos.x) / speed.x;
        }

        if (pos.y + speed.y < r) {
            ty = (r - pos.y) / speed.y;
        }
        else if (pos.y + speed.y > HEIGHT - r) {
            ty = (HEIGHT - r - pos.y) / speed.y;
        }

        int dir;
        double t;

        if (tx < ty) {
            dir = HORIZONTAL;
            t = tx + from;
        }
        else {
            dir = VERTICAL;
            t = ty + from;
        }

        if (t <= 0.0 || t > 1.0) {
            return INFINITY;
        }

        return t;// collisionsCache[collisionsCacheFE++]->update(t, this, dir);
    }

    virtual /*Collision**/double collision(Sim &u, double from) {
        double x2 = pos.x - u.pos.x;
        double y2 = pos.y - u.pos.y;
        double r2 = r + u.r;
        double vx2 = speed.x - u.speed.x;
        double vy2 = speed.y - u.speed.y;
        double a = vx2 * vx2 + vy2 * vy2;

        if (a < E) {
            return INFINITY;
        }

        double b = -2.0 * (x2 * vx2 + y2 * vy2);
        double delta = b * b - 4.0 * a * (x2 * x2 + y2 * y2 - r2 * r2);

        if (delta < 0.0) {
            return INFINITY;
        }

        // double sqrtDelta = sqrt(delta);
        // double d = 1.0/(2.0*a);
        // double t1 = (b + sqrtDelta)*d;
        // double t2 = (b - sqrtDelta)*d;
        // double t = t1 < t2 ? t1 : t2;

        double t = (b - sqrt(delta)) * (1.0 / (2.0 * a));

        if (t <= 0.0) {
            return INFINITY;
        }

        t += from;

        if (t > 1.0) {
            return INFINITY;
        }

        return t;//collisionsCache[collisionsCacheFE++]->update(t, this, u);
    }

    virtual void bounce(Sim &u) {
        double mcoeff = (m + u.m) / (m * u.m);
        double nx = pos.x - u.pos.x;
        double ny = pos.y - u.pos.y;
        double nxnydeux = nx * nx + ny * ny;
        double dvx = speed.x - u.speed.x;
        double dvy = speed.y - u.speed.y;
        double product = (nx * dvx + ny * dvy) / (nxnydeux * mcoeff);
        double fx = nx * product;
        double fy = ny * product;
        double m1c = 1.0 / m;
        double m2c = 1.0 / u.m;

        speed.x -= fx * m1c;
        speed.y -= fy * m1c;
        u.speed.x += fx * m2c;
        u.speed.y += fy * m2c;

        // Normalize vector at 100
        double impulse = sqrt(fx * fx + fy * fy);
        if (impulse < 100.0) {
            double min = 100.0 / impulse;
            fx = fx * min;
            fy = fy * min;
        }

        speed.x -= fx * m1c;
        speed.y -= fy * m1c;
        u.speed.x += fx * m2c;
        u.speed.y += fy * m2c;
    }

    virtual void bounce(int dir) {
        if (dir == HORIZONTAL) {
            speed.x = -speed.x;
        }
        else {
            speed.y = -speed.y;
        }
    }

    virtual void end(double fr) {
        pos.x = round(pos.x);
        pos.y = round(pos.y);
        speed.x = round(speed.x * fr);
        speed.y = round(speed.y * fr);
    }

/*  if (type == SNAFFLE) {
            return !carrier && !dead && !u.snaffle && !u.grab;
        }
        else if (u.type == SNAFFLE) {
            return !u.carrier && !u.dead && !snaffle && !grab;
        }

        return true;
    }*/


    virtual void print() {}

    virtual void save() {
        sx = pos.x;
        sy = pos.y;
        svx = speed.x;
        svy = speed.y;
    }

    virtual void reset() {
        pos.x = sx;
        pos.y = sy;
        speed.x = svx;
        speed.y = svy;
    }
        
};

struct Magic{
    int id_snaffle_magic_flip = -1;
    int id_snaffle_magic_acci = -1;
    int id_snaffle_magic_petr = -1;
    int turn_flip = 0;
    int turn_acci = 0;
    int turn_petr = 0;
    int type = -1;
    int simul = 0;
    int nid_snaffle_magic_flip = -1;
    int nid_snaffle_magic_acci = -1;
    int nid_snaffle_magic_petr = -1;
    int nturn_flip = 0;
    int nturn_acci = 0;
    int nturn_petr = 0;
};

class Node{
public:
    vector<Node*> child;
    Node *parent=nullptr;
    double par_score = 0;
    double ucb=0;
    double n=0;
    double w=0;
    int num = 0;
    ll score=0;
    bool terminal = false;
    string ans;
    string dir;
    Sim player;
    Sim playert2;
    Sim playert3;
    Sim player1;
    Sim player2;
    Sim player3;
    int choose_son;
    int depth=0;
    bool expand = false;
    long double variance=  0.0;
    long double mean = 0.0;
    string path;
    double highest = -500000000;
    double lowest = 500000000;
    int num_root = 0;
    vector<Sim> Opp;
    vector<Sim> Snaffle;
    vector<Sim> Bludgers;
    vector<bool> vissn;
    int scoret=0;
    int opp_score=0;
    int pid_snaffle_p1;
    int pid_snaffle_p2;
    int pt_magic, pt_magico;
    Magic magic, magic2, magic3, magic4;
    int win = -1;

    Node(){};
};

struct CustomCompare {
    bool operator()(const shared_ptr<Node> lhs, const shared_ptr<Node> rhs) const {
        return lhs->score > rhs->score;
    }
};

class Solutionm {
public:
    Sim moves1[20];
    Sim moves2[20];
    ll score;
    int tsort[5] = {-1, -1, -1, -1, -1};
    int tid_sort[5] = {-1, -1, -1, -1, -1};
    int count_sort[5] = {-1, -1, -1, -1, -1};
    int tsort2[5] = {-1, -1, -1, -1, -1};
    int tid_sort2[5] = {-1, -1, -1, -1, -1};
    int count_sort2[5] = {-1, -1, -1, -1, -1};

};



class Simulation{
public:
    int NB_SOL;
    int DEPTH;
    int tsort[5] = {-1, -1, -1, -1, -1};
    int tid_sort[5] = {-1, -1, -1, -1, -1};
    int count_sort[5] = {-1, -1, -1, -1, -1};
    int tsort2[5] = {-1, -1, -1, -1, -1};
    int tid_sort2[5] = {-1, -1, -1, -1, -1};
    int count_sort2[5] = {-1, -1, -1, -1, -1};
    int ITER = 0;
    int id_snaffle_p1 = -1, id_snaffle_p2 = -1, id_snaffle_p11 = -1, id_snaffle_p22 = -1;
    int ind_id_snaffle_p22 = -1;
    int place_sd1 = -1, place_sd2 = -1;
    int ball_rem = 0;
    int targetb1 = -1, targetb2=-1;
    int ind_throw1=-1, ind_throw2 = -1;
    vector<vector<Vector2>>dcell;
    Magic mp1, mp2, nmp1, nmp2;
    

    double mins = 10.0;

    vector<vector<double>> sortiesbs = {

        {0.0, 150.0},
        {45.0, 150.0},   // sortie 1
        {90.0, 150.0},     // sortie 2
        {135.0, 150.0},    // sortie 
        {180.0, 150.0}, // sortie 4
        {225.0, 150.0},   // sortie 5
        {270.0, 150.0},   // sortie 6
        {315.0, 150.0}   // sortie 6*/


    };
    

    vector<vector<double>> sortiesn = //{0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0};
    {
        /*{0.0, 490.0},
        {45.0, 490.0},   // sortie 1
        {90.0, 490.0},     // sortie 2
        {135.0, 490.0},    // sortie 
        {180.0, 490.0}, // sortie 4
        {225.0, 490.0},   // sortie 5
        {270.0, 490.0},   // sortie 6
        {315.0, 490.0},   // sortie 6*/
        {0.0, 500.0},
        {45.0, 500.0},   // sortie 1
        {90.0, 500.0},     // sortie 2
        {135.0, 500.0},    // sortie 
        {180.0, 500.0}, // sortie 4
        {225.0, 500.0},   // sortie 5
        {270.0, 500.0},   // sortie 6
        {315.0, 500.0}   // sortie 6
    };

    vector<vector<double>> sortiesnd = //{0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0};
    {
        /*{0.0, 0.0},
        {45.0, 0.0},   // sortie 1
        {90.0, 0.0},     // sortie 2
        {135.0, 0.0},    // sortie 
        {180.0, 0.0}, // sortie 4
        {225.0, 0.0},   // sortie 5
        {270.0, 0.0},   // sortie 6
        {315.0, 0.0},   // sortie 6*/
        {0.0, 175.0},
        {45.0, 175.0},   // sortie 1
        {90.0, 175.0},     // sortie 2
        {135.0, 175.0},    // sortie 
        {180.0, 175.0}, // sortie 4
        {225.0, 175.0},   // sortie 5
        {270.0, 175.0},   // sortie 6
        {315.0, 175.0}   // sortie 6
    };

    vector<Solutionm> solution ;
    Simulation(int nbs, int d):NB_SOL(nbs), DEPTH(d){
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dthrust(0, 150);
        std::uniform_int_distribution<int> dthrustt(0, 500);
        std::uniform_int_distribution<int> dx(0, 16000);
        std::uniform_int_distribution<int> dy(0, 7500);
        std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);

        

    }
   

    void Simulate(Sim& sim) {
        
        double acc = (double)sim.thrust;
        //cerr << "acc " << acc << endl;

       /* Vector2 dir{sim.x - sim.pos.x, sim.y - sim.pos.y};
        //cerr << "dir " << dir.x << " " << dir.y << endl;
        double nr = norme1(dir);
        dir.x /= nr;
        dir.y /= nr;

        dir.x *= acc;
        dir.y *= acc;*/

        Vector2 dir;
        dir.x = cos(sim.angle) * acc;
        dir.y = sin(sim.angle) * acc;

        sim.speed.x += dir.x;
        sim.speed.y += dir.y;

        sim.pos.x = (sim.pos.x+sim.speed.x);
        sim.pos.y = (sim.pos.y+sim.speed.y);

        sim.speed.x = round(sim.speed.x * 0.75);
        sim.speed.y = round(sim.speed.y * 0.75);


    }

    void Simulates(Sim& sim, Vector2 &poss) {
        
        double acc = (double)sim.thrustt / 0.5;
        //cerr << "acc " << acc << endl;

        /*Vector2 dir{sim.xt - sim.pos.x, sim.yt - sim.pos.y};
        //cerr << "dir " << dir.x << " " << dir.y << endl;
        double nr = norme1(dir);
        dir.x /= nr;
        dir.y /= nr;

        dir.x *= acc;
        dir.y *= acc;*/

        Vector2 dir;
        dir.x = cos(sim.anglet) * acc;
        dir.y = sin(sim.anglet) * acc;

        Vector2 speed = sim.speed;
        speed.x += dir.x;
        speed.y += dir.y;

        poss.x = (sim.pos.x+speed.x);
        poss.y = (sim.pos.y+speed.y);

          


    }
    
    float CollisionTime(Sim& p1, Sim& p2)
    {
        const Vector2 dP{p2.pos.x - p1.pos.x, p2.pos.y - p1.pos.y};
        const Vector2 dS{p2.speed.x - p1.speed.x, p2.speed.y - p1.speed.y};

        constexpr float eps = 0.000001f; // float precision...

        // we're looking for t such that:
        // |            p2(t)           -           p1(t)           | < 2*podRadius
        // |(p2.position + t*p2.speed)  - (p1.position + t*p1.speed)| < 2*podRadius
        // |(p2.position - p1.position) -  t*(p2.speed - p1.speed)  | < 2*podRadius
        // |         dP                 +            t*dS           | < 2*podRadius
        // t^2 dS^2 + t 2dPdS + dP^2 < 4 podRadius^2
        // t^2  a   + t   b   +   c      = 0;

        const float a = dot(dS, dS);
        if (a < eps) // moving away from each other
            return INFINITY;

        const float b = -2.f*dot(dP, dS);
        const float c = dot(dP,dP) - 4.f*(p1.radius*p2.radius);

        const float delta = b*b - 4.f*a*c;
        if (delta < 0.f) // no solution
            return INFINITY;

        const float t = (b - sqrt(delta)) / (2.f * a);
        if (t <= eps)
            return INFINITY;

        return t;
    }

    double Col(Sim p3, Sim pod2){
        double col = double((p3.speed.x * (pod2.pos.x - p3.pos.x) + p3.speed.y*(pod2.pos.y-p3.pos.y))) / 
                        double(sqrt(p3.speed.x*p3.speed.x + p3.speed.y*p3.speed.y) * 
                            sqrt((pod2.pos.x - p3.pos.x)* (pod2.pos.x - p3.pos.x) +  (pod2.pos.y - p3.pos.y) * (pod2.pos.y - p3.pos.y))+0.000001);

        return col;
    }
    
    void Simulate_Sorcier(Sim &sim, double mass, double cf){


        double acc = (double)sim.thrust / mass;
      
        Vector2 dir;
        dir.x = cosAngles[(int)sim.angle] * acc;
        dir.y = sinAngles[(int)sim.angle] * acc;

        sim.speed.x += dir.x;
        sim.speed.y += dir.y;

        

        //sim.speed.x = round(sim.speed.x * cf);
        //sim.speed.y = round(sim.speed.y * cf);

    
    }

    void Simulate_SorcierS(Sim &sim, double mass, double cf){


        double acc = (double)sim.thrust / mass;
      
        Vector2 dir;
        dir.x = cosAngles[(int)sim.angle] * acc;
        dir.y = sinAngles[(int)sim.angle] * acc;

        sim.speed.x += dir.x;
        sim.speed.y += dir.y;
      
        sim.pos.x += sim.speed.x;
        sim.pos.y += sim.speed.y;

        sim.pos.x = round(sim.pos.x);
        sim.pos.y = round(sim.pos.y);
        sim.speed.x = round(sim.speed.x * cf);
        sim.speed.y = round(sim.speed.y * cf);

        /*if(sim.pos.x <0)sim.pos.x = 0;
        if(sim.pos.x >16000)sim.pos.x = 16000;
        if(sim.pos.y <0)sim.pos.y = 0;
        if(sim.pos.y >7500)sim.pos.y = 7500;*/

    
    }

    void Update_pos(Sim &sim, double t){
        sim.pos.x = (sim.pos.x+sim.speed.x*t);
        sim.pos.y = (sim.pos.y+sim.speed.y*t);

        /*if(sim.pos.x <350)sim.pos.x = 0;
        if(sim.pos.x >16000-350)sim.pos.x = 16000;
        if(sim.pos.y <0)sim.pos.y = 0;
        if(sim.pos.y >7500)sim.pos.y = 7500;*/
    }


    int Selection(Node* root, Node** leaf, double scale){

        //std::mt19937 rng(std::random_device{}());
        //std::uniform_int_distribution<int> dexplor(1, 4);

        //long double explora = dexplor(rng);
        Node *node = root;
        int depth=0;

        
       
            
            for(int i = 0;i < node->child.size();++i){
                              
                //UCT
                if(node->child[i]->n != 0){
                    double ad = 0;
                    ad = sqrt(2.0*log(node->n) / node->child[i]->n);
                    node->child[i]->ucb = (long double)node->child[i]->score / (node->child[i]->n) + ad;
                    //cerr << node->child[i]->ucb << endl;
                }
                else{
                    node->child[i]->ucb = std::numeric_limits<long double>::max();
                }

                                              
                this->ITER++;

            }
            
            long double max_ucb= std::numeric_limits<double>::lowest();
            int ind = -1;
            for(int i = 0;i < node->child.size();++i){
            
                if(node->child[i]->ucb > max_ucb){
                    max_ucb = node->child[i]->ucb;
                    ind = i;
                }

                this->ITER++;
            }

    
                node = node->child[ind];
                depth++;
         

        *leaf = node;
        //cerr << "node="<< node << endl;
                
        return depth;


    }

    void Expand(Node *node, int depth){

        
        //direction
        for(int i = 0;i < 36;++i){
                        
            node->expand = true;
            Node *n = new Node();
            n->parent = node;
            n->depth = depth+1;
            n->num = i;
            this->ITER++;
            node->child.push_back(n);

        }

    }

    void ExpandS(Node *node, int depth){

        
        //direction
        for(int i = 0;i < 8;++i){
                        
            node->expand = true;
            Node *n = new Node();
            n->parent = node;
            n->depth = depth+1;
            n->num = i;
            this->ITER++;
            node->child.push_back(n);

        }

    }

    void Backpropagation(Node* node, double sc){

        // Backpropagation du score
        Node* par = node;
    
                        
        while(par != nullptr) {
            par->n++;  
            par->score += sc; 
            par->w++;
            this->ITER++;
            par = par->parent;
        }


    }

    bool getIntersection(Vector2 P1, Vector2 d1, Vector2 P2, Vector2 d2, Vector2& intersection) {
        float determinant = d1.x * d2.y - d1.y * d2.x;

        // If the determinant is 0, the lines are parallel or coincident
        if (determinant == 0) {
            return false;
        }

        float t = ((P2.x - P1.x) * d2.y - (P2.y - P1.y) * d2.x) / determinant;

        // Calculate the intersection point
        intersection.x = P1.x + t * d1.x;
        intersection.y = P1.y + t * d1.y;

        return true;
    }

    string Take_Action(int np, Magic &magic, bool take, bool &taken, Sim &p, Sim &p2, vector<Sim> &opp, vector<Sim> &snaffle, vector<Sim> &bludgers,  int maxball, int scoret, int &pt_magic,int pt_magico, int turn, int time, bool oppob){

        int nturns[5] = {0, 4,1 , 6, 3};

        //
        for(int i = 1;i <= 4;++i){
            if(tsort[i] != -1){
                count_sort[i]--;
                if(count_sort[i] == 0){
                    tsort[i] = -1;
                    count_sort[i] = nturns[i];
                }
            }

            if(tsort2[i] != -1){
                count_sort2[i]--;
                if(count_sort2[i] == 0){
                    tsort2[i] = -1;
                    count_sort2[i] = nturns[i];
                }
            }

        }

        Vector2 but;
        Vector2 cotev;
        int cote;
        if(p.idteam == 0){
            but = {16500, 3750};
            cotev = {-500, 3750};
            cote = 1;
        }
        else{
            but = {-500, 3750};
            cotev = {16500, 3750};
            cote = 2;
        }

        string ans;

        if(ans.empty() && pt_magic >= 15){
            double mind1 = 2000000.0, mind2 = 2000000.0;
            int io1=-1, io2=-1;
            for(int i = 0;i < snaffle.size();++i){
                //if(i == this->ind_throw1 || i == this->ind_throw2)continue;
           
                if(np == 2 && snaffle[i].id != this->id_snaffle_p2)continue;
                if(np == 1 && snaffle[i].id != this->id_snaffle_p1)continue;

                if(p.idteam == 0 && snaffle[i].pos.x > p.pos.x)continue;
                if(p.idteam == 1 && snaffle[i].pos.x < p.pos.x)continue;
             
                double d = distance(p.pos, snaffle[i].pos);
                if(d <= 2000 || d > 6000)continue;
                double d2 = distance(p2.pos, snaffle[i].pos);
                if(d2 <= 399)continue;

                double do1 = distance(snaffle[i].pos, opp[0].pos);
                double do2 = distance(snaffle[i].pos, opp[1].pos);

                if(do1 < mind1){
                    mind1 = do1;
                    io1 = i;
                }

                if(do2 < mind2){
                    mind2 = do2;
                    io2 = i;
                }
            

            }

            if(io1 != -1 && io2 != -1){
                int inda = (mind1 <= mind2)? io1 : io2; 
                ans="ACCIO " + to_string(snaffle[inda].id);
                pt_magic -= 15;

                
                magic.turn_acci = 6;
                magic.id_snaffle_magic_acci = snaffle[inda].id;
                magic.type = 1;
                

            }

        }


        /*if(ans.empty() && pt_magic >= 20){

            Vector2 _P21{16000, 2025};
            Vector2 _d21{0, 5415-2025};
            Vector2 _P22{0,2025};
            Vector2 _d22{0, 5415-2025};

            for(int i = 0;i < snaffle.size();++i){
                if(cote == 1 && p.pos.x > snaffle[i].pos.x)continue;
                if(cote == 2 && p.pos.x < snaffle[i].pos.x)continue;
                              
                Vector2 speed = {(snaffle[i].pos.x - p.pos.x)*1.0f, (snaffle[i].pos.y - p.pos.y)*1.0f};
                Sim lancer;
                lancer.speed = speed;
                lancer.pos = p.pos;
                Vector2 P1 = p.pos;
                Vector2 d1 = lancer.speed;
                Vector2 P2, d2;
                if(cote == 1){
                    P2 = _P21;
                    d2 = _d21;
                    
                }
                else{
                    P2 = _P22;
                    d2 = _d22;

                }

                double d = distance(p.pos, snaffle[i].pos);
                //if(d < 3000.0)continue;

                Vector2 intersection;

                bool inter = this->getIntersection(P1, d1, P2, d2, intersection);
                if(inter){
                    if (intersection.y <= 5415 && intersection.y >= 2025){
                        ans="FLIPENDO " + to_string(snaffle[i].id) + " AOUUUU!!!";
                        pt_magic -= 20;

                        
                        magic.turn_flip = 3;
                        magic.id_snaffle_magic_flip = snaffle[i].id;
                        magic.type = 0;
                       
                    
                        break;
                    }

                }

                

            }

          


        }*/

      
        if(p.use_all == 1){
            /*if(ans.empty() && pt_magic >= 15){
                double mind1 = 2000000.0, mind2 = 2000000.0;
                int io1=-1, io2=-1;
                for(int i = 0;i < snaffle.size();++i){
                    //if(i == this->ind_throw1 || i == this->ind_throw2)continue;
                    if(snaffle.size() > 1 && !oppob){
                        if(snaffle[i].id == this->id_snaffle_p2)continue;
                        if(snaffle[i].id == this->id_snaffle_p1)continue;
                    }
                    double d = distance(p.pos, snaffle[i].pos);
                    if(d <= 399 || d > 4000)continue;
                    double d2 = distance(p2.pos, snaffle[i].pos);
                    if(d2 <= 399)continue;

                    double do1 = distance(snaffle[i].pos, opp[0].pos);
                    double do2 = distance(snaffle[i].pos, opp[1].pos);

                    if(do1 < mind1){
                        mind1 = do1;
                        io1 = i;
                    }

                    if(do2 < mind2){
                        mind2 = do2;
                        io2 = i;
                    }
                

                }

                if(io1 != -1 && io2 != -1){
                    int inda = (mind1 <= mind2)? io1 : io2; 
                    ans="ACCIO " + to_string(snaffle[inda].id);
                    pt_magic -= 15;

                    
                    magic.turn_acci = 6;
                    magic.id_snaffle_magic_acci = snaffle[inda].id;
                    magic.type = 1;
                    

                }

            }*/
            
            if(ans.empty() && pt_magic >= 25 && snaffle.size() == 1){
                
                
                for(int i = 0;i < snaffle.size();++i){
                
                    double d = distance(snaffle[i].pos, cotev);
                    if(d < 2000){
                        ans="PETRIFICUS " + to_string(snaffle[i].id);
                        pt_magic -= 10;

                        
                        magic.id_snaffle_magic_petr = snaffle[i].id;
                        magic.type = 2;
                        magic.turn_petr = 1;
                       


                        //taken = true;
                        break;
                    }
                                    
                }
                                    
            

                

            }
        }
        /*if(ans.empty() && pt_magic >= 5 && snaffle.size() == 1 ){

            for(int i = 0;i < bludgers.size();++i){
                
                double d = distance(bludgers[i].pos, p.pos);
                if(d < 2000 && this->Col(bludgers[i], p)>= 0.9){
                    ans="OBLIVIATE " + to_string(bludgers[i].id);
                    pt_magic -= 5;
                   
                  
                    break;
                }
                                
            }

            

        }*/

        return ans;


    }

    string Take_ActionD(int np, Magic &magic, bool take, bool &taken, Sim &p, Sim &p2, vector<Sim> &opp, vector<Sim> &snaffle, vector<Sim> &bludgers,  int maxball, int scoret, int &pt_magic, int pt_magico, int turn, int time, bool oppob){

        int nturns[5] = {0, 4,1 , 6, 3};

        //
        for(int i = 1;i <= 4;++i){
            if(tsort[i] != -1){
                count_sort[i]--;
                if(count_sort[i] == 0){
                    tsort[i] = -1;
                    count_sort[i] = nturns[i];
                }
            }

            if(tsort2[i] != -1){
                count_sort2[i]--;
                if(count_sort2[i] == 0){
                    tsort2[i] = -1;
                    count_sort2[i] = nturns[i];
                }
            }

        }

        Vector2 but;
        Vector2 cotev;
        int cote;
        if(p.idteam == 0){
            but = {16500, 3750};
            cotev = {-500, 3750};
            cote = 1;
        }
        else{
            but = {-500, 3750};
            cotev = {16500, 3750};
            cote = 2;
        }

        string ans;

        if(ans.empty() && pt_magic >= 15){
            double mind1 = 2000000.0, mind2 = 2000000.0, minme = 2000000.0;
            int io1=-1, io2=-1;
            for(int i = 0;i < snaffle.size();++i){
                //if(i == this->ind_throw1 || i == this->ind_throw2)continue;
            
                if(np == 2 && snaffle[i].id != this->id_snaffle_p2)continue;
                if(np == 1 && snaffle[i].id != this->id_snaffle_p1)continue;

                if(p.idteam == 0 && snaffle[i].pos.x > p.pos.x)continue;
                if(p.idteam == 1 && snaffle[i].pos.x < p.pos.x)continue;
                
                double d = distance(p.pos, snaffle[i].pos);
                if(d <= 399 || d > 6000)continue;
                double d2 = distance(p2.pos, snaffle[i].pos);
                if(d2 <= 399)continue;

                double do1 = distance(snaffle[i].pos, opp[0].pos);
                double do2 = distance(snaffle[i].pos, opp[1].pos);

                if(d < minme){
                    minme = d;
                    
                }

                if(do1 < mind1){
                    mind1 = do1;
                    io1 = i;
                }

                if(do2 < mind2){
                    mind2 = do2;
                    io2 = i;
                }
            

            }

            if(mind1 < minme || mind2 < minme){
                int inda = (mind1 <= mind2)? io1 : io2; 
                ans="ACCIO " + to_string(snaffle[inda].id);
                pt_magic -= 15;

                
                magic.turn_acci = 6;
                magic.id_snaffle_magic_acci = snaffle[inda].id;
                magic.type = 1;
                

            }

        }

        
        /*if(ans.empty() && pt_magic >= 20){
            Vector2 _P21{16000, 2025};
            Vector2 _d21{0, 5415-2025};
            Vector2 _P22{0,2025};
            Vector2 _d22{0, 5415-2025};

            for(int i = 0;i < snaffle.size();++i){
                if(cote == 1 && p.pos.x > snaffle[i].pos.x)continue;
                if(cote == 2 && p.pos.x < snaffle[i].pos.x)continue;
                              
                Vector2 speed = {(snaffle[i].pos.x - p.pos.x)*1.0f, (snaffle[i].pos.y - p.pos.y)*1.0f};
                Sim lancer;
                lancer.speed = speed;
                lancer.pos = p.pos;
                Vector2 P1 = p.pos;
                Vector2 d1 = lancer.speed;
                Vector2 P2, d2;
                if(cote == 1){
                    P2 = _P21;
                    d2 = _d21;
                    
                }
                else{
                    P2 = _P22;
                    d2 = _d22;

                }

                double d = distance(p.pos, snaffle[i].pos);
                if(d < 3000.0)continue;

                Vector2 intersection;

                bool inter = this->getIntersection(P1, d1, P2, d2, intersection);
                if(inter){
                    if (intersection.y <= 5415 && intersection.y >= 2025){
                        ans="FLIPENDO " + to_string(snaffle[i].id) + " AOUUUU!!!";
                        pt_magic -= 20;

                        
                        magic.turn_flip = 3;
                        magic.id_snaffle_magic_flip = snaffle[i].id;
                        magic.type = 0;
                        
                    
                        break;
                    }

                }

                

            }

          


        }*/

        /*if(ans.empty() && pt_magic >= 20 && snaffle.size() == 1){
           
            int d = distance(p.pos, snaffle[0].pos);
            int d2 = distance(opp[0].pos, snaffle[0].pos);
            int d3 = distance(opp[1].pos, snaffle[0].pos);

            if( d2 < d || d3 < d){

                int ind = 0;
                if(d3 < d2)ind = 1;

                ans="FLIPENDO " + to_string(opp[ind].id) + " AOUKIUSHUUUUU!!!";
                pt_magic -= 20;

                
                magic.turn_flip = 3;
                magic.id_snaffle_magic_flip = opp[ind].id;
                magic.type = 1;

            }
                             


        }*/
        
        if(p.use_all ==  1){
            /*if(ans.empty() && pt_magic >= 15){
                double mind1 = 2000000.0, mind2 = 2000000.0;
                int io1=-1, io2=-1;

                for(int i = 0;i < snaffle.size();++i){
            
                    //if(i == this->ind_throw1 || i == this->ind_throw2)continue;
                    if(snaffle.size() > 1 && !oppob){
                        if(snaffle[i].id == this->id_snaffle_p2)continue;
                        if(snaffle[i].id == this->id_snaffle_p1)continue;
                    }

                    double d = distance(p.pos, snaffle[i].pos);
                    if(d <= 399 || d > 4000)continue;
                    double d2 = distance(p2.pos, snaffle[i].pos);
                    if(d2 <= 399)continue;

                    double do1 = distance(snaffle[i].pos, opp[0].pos);
                    double do2 = distance(snaffle[i].pos, opp[1].pos);

                    if(do1 < mind1){
                        mind1 = do1;
                        io1 = i;
                    }

                    if(do2 < mind2){
                        mind2 = do2;
                        io2 = i;
                    }
                

                }

                if(io1 != -1 && io2 != -1){
                    int inda = (mind1 <= mind2)? io1 : io2; 
                    ans="ACCIO " + to_string(snaffle[inda].id);
                    pt_magic -= 15;

                    
                    magic.turn_acci = 6;
                    magic.id_snaffle_magic_acci = snaffle[inda].id;
                    magic.type = 1;
                   

                }

                

            }*/
            
            if(ans.empty() && pt_magic >= 25){
                Vector2 _P21{16000, 2025};
                Vector2 _d21{0, 5700-1700};
                Vector2 _P22{0,2025};
                Vector2 _d22{0, 5700-1700};
    
                for(int i = 0;i < snaffle.size();++i){
                    if(cote == 1 && p.pos.x < snaffle[i].pos.x)continue;
                    if(cote == 2 && p.pos.x > snaffle[i].pos.x)continue;
                    if( snaffle[i].speed.x == 0 &&  snaffle[i].pos.y == 0)continue;
    
                    Vector2 P1 = snaffle[i].pos;
                    Vector2 d1 = snaffle[i].speed;
                    Vector2 P2, d2;
                    if(cote == 1){
                        P2 = _P22;
                        d2 = _d22;
                        
                    }
                    else{
                        P2 = _P21;
                        d2 = _d21;
    
                    }
    
                    
    
                    Vector2 intersection;
    
                    bool inter = this->getIntersection(P1, d1, P2, d2, intersection);
    
                    Sim si;
                    si.pos = intersection;
                    si.speed = {0.0f, 0.0f};
                    //float t = this->CollisionTime(p, si);
                
                    double d = distance(snaffle[i].pos, cotev);
                    if(inter && (intersection.y <= 5415 && intersection.y >= 2025)){
                        ans="PETRIFICUS " + to_string(snaffle[i].id);
                        pt_magic -= 10;
                    
                     
                        magic.turn_petr = 1;
                        magic.id_snaffle_magic_petr = snaffle[i].id;
                        magic.type = 2;
                        
    
                        //taken = true;
                        break;
                    }
                                    
                }
       
    
            }
            
            

        }

        
        
        /*if(ans.empty() && pt_magic >= 5 && snaffle.size() == 1){

            for(int i = 0;i < bludgers.size();++i){
                
                double d = distance(bludgers[i].pos, p.pos);
                if(d < 2000 && this->Col( bludgers[i], p)>= 0.9){
                    ans="OBLIVIATE " + to_string(bludgers[i].id);
                    pt_magic -= 5;
                  
                    break;
                }
                                
            }

            

        }*/

        return ans;


    }

    double GetAngle(Vector2 pos){

        float angle_radians = atan2(pos.y, pos.x);
        float angle_degrees = angle_radians * (180.0f / M_PI);

        // Si l'angle est négatif, on ajoute 360 pour le ramener entre 0 et 360 degrés
        if (angle_degrees < 0) {
            angle_degrees += 360.0f;
        }

        return angle_degrees;

    }

    Vector2 PrepaAstar(Sim p, Sim o2, Sim o3, Sim o4, Sim o5, Sim o6, bool throws, int num){
        vector<vector<bool>>vis(dcell.size(), vector<bool>(dcell[0].size(), false));

        double mind = 2000000.0, mind2 = 2000000.0, mind3 = 2000000.0, mind4 = 2000000.0, mind5 = 2000000.0, mind6 = 2000000.0;
        Vector2 start, start2, start3, start4, start5, start6;
        

        for(int i = 0;i < dcell.size();++i){
            for(int j = 0;j < dcell[0].size();++j){
                if(i == start.y && j == start.x)continue;
                if(i == VEND1.y && j == VEND1.x)continue;
                
                double d2 = distance(dcell[i][j], o2.pos);
                if(d2 < mind2){
                    mind2 = d2;
                    start2.x = j;
                    start2.y = i;
                }

                if(!throws && d2 <= 400){
                    vis[i][j] = true;
                }

                double d3 = distance(dcell[i][j], o3.pos);
                if(d3 < mind3){
                    mind3 = d3;
                    start3.x = j;
                    start3.y = i;
                }

                if(d3 <= 400){
                    vis[i][j] = true;
                }

                double d4 = distance(dcell[i][j], o4.pos);
                if(d4 < mind4){
                    mind4 = d4;
                    start4.x = j;
                    start4.y = i;
                }

                if(d4<= 400){
                    vis[i][j] = true;
                }

                double d5 = distance(dcell[i][j], o5.pos);
                if(d5 < mind5){
                    mind5 = d5;
                    start5.x = j;
                    start5.y = i;
                }

                if(d5 <= 400){
                    vis[i][j] = true;
                }

                double d6 = distance(dcell[i][j], o6.pos);
                if(d6 < mind6){
                    mind6 = d6;
                    start6.x = j;
                    start6.y = i;
                }

                if(d6 <= 400){
                    vis[i][j] = true;
                }

            }
        }

        for(int i = 0;i < dcell.size();++i){
            for(int j = 0;j < dcell[0].size();++j){
                double d = distance(dcell[i][j], p.pos);
                if(d < mind){
                    mind = d;
                    start.x = j;
                    start.y = i;
                }

                if(d <= 400){
                    vis[i][j] = false;
                }
               
            }
        }
    
        
        /*if(!throws)vis[(int)start2.y][(int)start2.x] = true;
        vis[(int)start3.y][(int)start3.x] = true;
        vis[(int)start4.y][(int)start4.x] = true;
        vis[(int)start5.y][(int)start5.x] = true;
        vis[(int)start6.y][(int)start6.x] = true;*/

        return astar(start, VEND1, vis, num);  


    }

    
    Vector2 astar(Vector2 start, Vector2 end, vector<vector<bool>>&vis, int num){

        priority_queue<pair<double, pair<Vector2, vector<Vector2>>>, vector<pair<double, pair<Vector2, vector<Vector2>>>>, Compare> minq;
        

        pair<double, pair<Vector2, vector<Vector2>>> p;

        minq.push({0, {start, {}} });

        vector<vector<int>> coord = {{-1, -1}, {0, -1}, {1, -1},
                                    {-1, 0}, {0, 1},
                                    {-1, 1}, {0, 1}, {1, 1}};

        cerr << "start " << start.x << " " << start.y << endl;
        cerr << "end " << end.x << " " << end.y << endl;

        while(!minq.empty()){

            p = minq.top();
            minq.pop();

            int costcurrent = p.first;
            Vector2 current = p.second.first;
            vector<Vector2> pathcurrent = p.second.second;

            if(!vis[(int)current.y][(int)current.x]){

                vis[(int)current.y][(int)current.x] = true;
                pathcurrent.push_back(current);
                if(current.x == end.x && current.y == end.y){
                    num = min(num, (int)pathcurrent.size());
                    cerr <<"size="<< pathcurrent.size() << endl;
                    if(num <= 0)return {-1, -1};
                    Vector2 f = pathcurrent[num];
                    return dcell[(int)f.y][(int)f.x];
                }

                for(int i = 0;i < 8;++i){
                    int x = current.x + coord[i][0];
                    int y = current.y + coord[i][1];
                    if(x < 0 || x>= dcell[0].size() || y < 0 || y>=dcell.size())continue;
                    
                    double cost = distance(dcell[(int)current.y][(int)current.x], dcell[y][x]);
                    minq.push({costcurrent+cost, {Vector2{(float)x, (float)y}, pathcurrent}});

                }


            }


        }

        return {-1, -1};


    }

    // Distance entre un point et un segment
    double distancePointToSegment(const Vector2& p, const Vector2& a, const Vector2& b) {
        double l2 = (b - a).lengthSquared();
        if (l2 == 0.0) return (p - a).length();
        double t = max(0.0, min(1.0, dot(p - a, b - a) / l2));
        Vector2 projection = a + (b - a) * t;
        return (p - projection).length();
    }

    // Convertit angle + distance en coordonnées
    Vector2 positionTir(const Vector2& currentPosition, double angleDeg, double distance) {
        double angleRad = angleDeg * M_PI / 180.0;
        return {
            currentPosition.x + cos(angleRad) * distance,
            currentPosition.y + sin(angleRad) * distance
        };
    }

    // Vérifie si un tir est dégagé
    bool isTrajectoryClear(const Vector2& start, const Vector2& end, const vector<Sim>& opponents, double clearance) {
        for (const auto& opponent : opponents) {
            double dist = distancePointToSegment(opponent.pos, start, end);
            if (dist < clearance) return false;
        }
        return true;
    }

    Vector2 bestShot(const Vector2& myPos, const vector<Sim>& opponents, const Vector2& but) {
        vector<vector<double>> sortiesbs = {
            {0.0, 500.0}, {45.0, 500.0}, {90.0, 500.0}, {135.0, 500.0},
            {180.0, 500.0}, {225.0, 500.0}, {270.0, 500.0}, {315.0, 500.0}
        };
    
        double clearance = 600.0;
        Vector2 bestTir = but;
        double bestAngleDiff = numeric_limits<double>::max();
    
        Vector2 dirBut = but - myPos;
    
        for (auto& sortie : sortiesbs) {
            Vector2 target = positionTir(myPos, sortie[0], 500.0);
    
            if (!isTrajectoryClear(myPos, target, opponents, clearance))
                continue;
    
            Vector2 tirVec = target - myPos;
    
            // Calculer l'angle entre tirVec et dirBut
            double dotProduct = dot(tirVec, dirBut);
            double lenProduct = tirVec.length() * dirBut.length();
    
            if (lenProduct == 0.0) continue;
    
            double angleCos = dotProduct / lenProduct; // entre -1 et 1
            double angleDiff = acos(angleCos);         // plus c’est petit, plus c’est aligné
    
            if (angleDiff < bestAngleDiff) {
                bestAngleDiff = angleDiff;
                bestTir = target;
            }
        }
    
        return bestTir;
    }

    bool snaffle_exist(int id, vector<Sim> snaffle){
        for(int i = 0;i < snaffle.size();++i){
            if(snaffle[i].id == id)return true;
        }
        return false;
    }

    int snaffle_id(int id, vector<Sim> snaffle){
        for(int i = 0;i < snaffle.size();++i){
            if(snaffle[i].id == id)return i;
        }
        return 0;
    }

    void set_magic(Magic &m, vector<Sim> snaffle){

        if(!snaffle_exist(m.id_snaffle_magic_flip, snaffle)){
            m.id_snaffle_magic_flip = -1;
            m.turn_flip = 0;
        }

        if(!snaffle_exist(m.id_snaffle_magic_acci, snaffle)){
            m.id_snaffle_magic_acci = -1;
            m.turn_acci = 0;
        }

        if(!snaffle_exist(m.id_snaffle_magic_petr, snaffle)){
            m.id_snaffle_magic_petr = -1;
            m.turn_petr = 0;
        }

        if(m.turn_flip > 0){
            m.turn_flip--;
            
        }
        
        if(m.turn_acci > 0){
            m.turn_acci--;
            
        }
        
        if(m.turn_petr > 0){
            m.turn_petr--;
          
        }



    }

    bool Get_Magic(int indpl, Sim &player, Sim &player2, Magic &magic, int opp_score, 
        vector<Sim> &Opp, vector<Sim> &Snaffle, vector<Sim> &Bludgers,  int maxball, int scoret, int &pt_magic,int pt_magico, int turn, int time, bool oppob){

        bool is_magic = false;
        bool take , taken;
        //magic
        if(player.state == 0 && ( (player.idteam == 0 && player.pos.x > player2.pos.x) || (player.idteam == 1 && player.pos.x < player2.pos.x) ) ){
            bool take = true, taken = false;
            if(scoret == this->ball_rem-1 || (opp_score - scoret) >= 2 || opp_score == this->ball_rem-1)player.use_all = 1;
         
            string magic_s1 = this->Take_Action(indpl+1, magic, take, taken, player, player2, Opp, Snaffle, Bludgers, maxball, scoret, pt_magic,pt_magico,  turn, time, oppob);

            if(!magic_s1.empty())is_magic = true;

            
            
        }
        else if(player.state == 0  && ( (player.idteam == 0 && player.pos.x < player2.pos.x) || (player.idteam == 1 && player.pos.x > player2.pos.x) )){
            bool take = true, taken = false;
            if(scoret == this->ball_rem-1 || (opp_score - scoret) >= 2 || opp_score == this->ball_rem-1)player.use_all = 1;
          
            string magic_s1 = this->Take_ActionD(indpl+1, magic, take, taken, player, player2, Opp, Snaffle, Bludgers, maxball, scoret, pt_magic,pt_magico, turn, time, oppob);

            if(!magic_s1.empty())is_magic = true;
            
        }

        if(magic.turn_flip > 0){
            magic.turn_flip--;
            is_magic = true;
        }
        
        if(magic.turn_acci > 0){
            magic.turn_acci--;
            is_magic = true;
        }
        
        if(magic.turn_petr > 0){
            magic.turn_petr--;
            is_magic = true;
        }
        
        return is_magic;


    }

    void Apply_Magic(Sim &player, Magic &magic, vector<Sim> &Snaffle){

        if(magic.turn_flip > 0){
            if(magic.id_snaffle_magic_flip == -1)cerr << "1 -1" << endl;
            int id = snaffle_id(magic.id_snaffle_magic_flip, Snaffle);
            double dx = Snaffle[id].pos.x - player.pos.x;
            double dy = Snaffle[id].pos.y - player.pos.y;
            double angleRadians = std::atan2(dy, dx);
            double angleDegres = angleRadians * 180.0 / M_PI;
            if(angleDegres < 0)angleDegres += 360;
            Snaffle[id].angle = angleDegres;
            double Dist = distance(Snaffle[id].pos, player.pos);
            Snaffle[id].thrust = min( 6000.0 / (( Dist / 1000.0 )*( Dist / 1000.0 )), 1000.0 );
            this->Simulate_Sorcier(Snaffle[id], 0.5, 0.75);
           
        }
        if(magic.turn_acci > 0){
            if(magic.id_snaffle_magic_acci == -1)cerr << "2 -1" << endl;;
            int id = snaffle_id(magic.id_snaffle_magic_acci, Snaffle);
            double dx = player.pos.x -Snaffle[id].pos.x;
            double dy = player.pos.y - Snaffle[id].pos.y;
            double angleRadians = std::atan2(dy, dx);
            double angleDegres = angleRadians * 180.0 / M_PI;
            if(angleDegres < 0)angleDegres += 360;
            Snaffle[id].angle = angleDegres;
            double Dist = distance(Snaffle[id].pos, player.pos);
            Snaffle[id].thrust = min( 3000.0 / (( Dist / 1000.0 )*( Dist / 1000.0 )), 1000.0 );
            this->Simulate_Sorcier(Snaffle[id], 0.5, 0.75);
            //this->Update_pos(Snaffle[id_snaffle_magic], 1.0);
            //Snaffle[id_snaffle_magic].end(0.75);

            
        }
        if(magic.turn_petr > 0){
            if(magic.id_snaffle_magic_petr == -1)cerr << "3 -1" << endl;;
            int id = snaffle_id(magic.id_snaffle_magic_petr, Snaffle);
            Snaffle[id].speed.x = 0;
            Snaffle[id].speed.y = 0;
            
                                        
        }

    }

    Vector2 getBorderIntersection(Vector2 P, Vector2 D) {
        float min_t = 1e9;
        Vector2 contact;
    
        // Bord haut (y = 0)
        if (D.y != 0) {
            float t = (0 - P.y) / D.y;
            float x = P.x + D.x * t;
            if (t > 0 && x >= 0 && x <= 16000 && t < min_t) {
                min_t = t;
                contact = {x, 0};
            }
        }
    
        // Bord bas (y = 7500)
        if (D.y != 0) {
            float t = (7500 - P.y) / D.y;
            float x = P.x + D.x * t;
            if (t > 0 && x >= 0 && x <= 16000 && t < min_t) {
                min_t = t;
                contact = {x, 7500};
            }
        }
    
        // Bord gauche (x = 0)
        if (D.x != 0) {
            float t = (0 - P.x) / D.x;
            float y = P.y + D.y * t;
            if (t > 0 && y >= 0 && y <= 7500 && t < min_t) {
                min_t = t;
                contact = {0, y};
            }
        }
    
        // Bord droit (x = 16000)
        if (D.x != 0) {
            float t = (16000 - P.x) / D.x;
            float y = P.y + D.y * t;
            if (t > 0 && y >= 0 && y <= 7500 && t < min_t) {
                min_t = t;
                contact = {16000, y};
            }
        }
    
        return contact;
    }
    

    string BeamSearch(int indpl, Sim &p, Sim &p2, vector<Sim> &opp, vector<Sim> &snaffle, vector<Sim> &bludgers,  int maxball, int scoret, int pt_magic, int pt_magico, int turn, int time, int opp_score){

        auto startm = high_resolution_clock::now();;
        int maxt = -1;
        auto getTime = [&]()-> bool {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - startm);
            //cerr << duration.count() << endl;
            maxt = duration.count();
            return(duration.count() <= time);
        };

        Vector2 but, buto;
        Vector2 cote;
        int cage = -1;
        if(p.idteam == 0){
            but = {16500, 3750};
            buto = {-500, 3750};
            cote = {-500, 3750};
            cage = 1;
        }
        else{
            but = {-500, 3750};
            buto = {16500, 3750};
            cote = {16500, 3750};
            cage = 2;
        }

        if(indpl == 0){
            set_magic(nmp1, snaffle);
        }
        else{
            set_magic(nmp2, snaffle);
        }

        if(p.state == 0 && ( (p.idteam == 0 && p.pos.x > p2.pos.x) || (p.idteam == 1 && p.pos.x < p2.pos.x) ) ){
            bool take = true, taken = false;
            if(scoret == this->ball_rem-1 || (opp_score - scoret) >= 2 || opp_score == this->ball_rem-1)p.use_all = 1;
            //p.use_all = 1;
            //if(pt_magic <= 20 && snaffle.size() > 1)p.use_all = 0;
            string magic_s1;
            if(indpl == 0)
                magic_s1 = this->Take_Action(indpl+1, nmp1, take, taken, p, p2, opp, snaffle, bludgers, maxball, scoret, pt_magic, pt_magico, turn, time, false);
            else
                magic_s1 = this->Take_Action(indpl+1, nmp2, take, taken, p, p2, opp, snaffle, bludgers, maxball, scoret, pt_magic,pt_magico, turn, time, false);

            if(!magic_s1.empty())return magic_s1;
         
        }
        else if(p.state == 0  && ( (p.idteam == 0 && p.pos.x < p2.pos.x) || (p.idteam == 1 && p.pos.x > p2.pos.x) )){
            bool take = true, taken = false;
            if(scoret == this->ball_rem-1 || (opp_score - scoret) >= 2 || opp_score == this->ball_rem-1)p.use_all = 1;
            //p.use_all = 1;
            //if(pt_magic <= 20 && snaffle.size() > 1)p.use_all = 0;
      
            string magic_s1;
            if(indpl == 0)
                magic_s1 = this->Take_ActionD(indpl+1, nmp1, take, taken, p, p2, opp, snaffle, bludgers, maxball, scoret, pt_magic,pt_magico, turn, time, false);
            else
                magic_s1 = this->Take_ActionD(indpl+1, nmp2, take, taken, p, p2, opp, snaffle, bludgers, maxball, scoret, pt_magic,pt_magico, turn, time, false);

            if(!magic_s1.empty())return magic_s1;
         
        }
        else if(p.state == 1){
            vector<Sim> OPPO;
            OPPO.insert(OPPO.end(), opp.begin(), opp.end());
            OPPO.insert(OPPO.end(), bludgers.begin(), bludgers.end());

            Vector2 tir = bestShot(p.pos, OPPO, but);
            ///(tir.x != -1)tir = but;
            string ans =  "THROW " + to_string((int)tir.x) + " " +  to_string((int)tir.y) + " 500";
            return ans;
        }

        int ind_snaffle = -1;
        int ind_snaffle2 = -1;
        int throwsn = 0;

        if(!snaffle_exist(this->id_snaffle_p1, snaffle))this->id_snaffle_p1 = -1;
        if(!snaffle_exist(this->id_snaffle_p2, snaffle))this->id_snaffle_p2 = -1;
       

        int ind_thsn = -1;
        double mind1 = 200000;
        int indm1 = -1;
        for(int i = 0;i < snaffle.size();++i){
            if(indpl == 0 && snaffle[i].id == this->id_snaffle_p2 && snaffle.size() >1 )continue;
            if(indpl == 1 && snaffle[i].id == this->id_snaffle_p1 && snaffle.size() > 1)continue;
            
            //double dc = distance(opp[ind_chaseo].pos, snaffle[i].pos);
            double d = distance(snaffle[i].pos, p.pos);
            double d2 = distance(snaffle[i].pos, p2.pos);

            if(/*(ind_def == 1 || ind_def == -1)  &&*/ d < mind1 && d < d2){//aff
                mind1 = d;
                indm1 = i;
            }
            /*else if(ind_def == 0 && dc < mind1){//def
                mind1 = dc;
                indm1 = i;
            }*/

            

        }

        if(indm1 != -1){
            ind_snaffle = indm1;
            if(indpl == 0)this->id_snaffle_p1 = snaffle[ind_snaffle].id;
            if(indpl == 1)this->id_snaffle_p2 = snaffle[ind_snaffle].id;
            
                    
        }

        //cerr << "inddef=" << ind_def << " " << ind_chaseo << " "<< this->id_snaffle_p1 << " " << this->id_snaffle_p2 << endl;
     

        int MAX_WIDTH = 768;
        int depth = 0;

        shared_ptr<Node> root = make_shared<Node>();
        root->player = p;
        root->player2 = p2;
        root->score = 0;
        root->Snaffle = snaffle;
        root->Bludgers = bludgers;
        root->Opp = opp;
        root->pid_snaffle_p1 = id_snaffle_p1;
        root->pid_snaffle_p2 = id_snaffle_p2;
        root->scoret = scoret;
        root->opp_score = opp_score;
        root->vissn = vector<bool>(snaffle.size(), false);
        root->pt_magic = pt_magic;
        root->pt_magico = pt_magico;
        if(indpl == 0){
            root->magic = nmp1;
            root->magic2 = nmp2;
        }
        else{
            root->magic = nmp2;
            root->magic2 = nmp1;
        }

        Node T0 = *root;
        ll scoref = -2000000000;

        multiset<shared_ptr<Node>, CustomCompare> beam;

        beam.insert(root);

        
        Sim P2 = p2;

        bool end = false;

        while (getTime() && depth < 20) {

            //cerr << "startTime" << depth << endl;

            multiset<shared_ptr<Node>, CustomCompare> TF;
            multiset<shared_ptr<Node>, CustomCompare>::iterator itbeam = beam.begin();

            for (int W = 0;getTime() && TF.size() < MAX_WIDTH && itbeam != beam.end(); ++W) {
                shared_ptr<Node> t = *itbeam;
                itbeam++;

              
                for(int i = 0;getTime() && i < 8;++i){

                    shared_ptr<Node> song = make_shared<Node>();
                    song->player = t->player;
                    song->player2 = t->player2;
                    song->Snaffle = t->Snaffle;
                    song->Bludgers = t->Bludgers;
                    song->Opp = t->Opp;
                    song->pid_snaffle_p1 = t->pid_snaffle_p1;
                    song->pid_snaffle_p2 = t->pid_snaffle_p2;
                    song->scoret = t->scoret;
                    song->opp_score = t->scoret;
                    song->vissn = t->vissn;
                    song->pt_magic = t->pt_magic;
                    song->pt_magico = t->pt_magico;
                    song->magic = t->magic;
                    song->magic2 = t->magic2;
                    song->magic3 = t->magic3;
                    song->magic4 = t->magic4;
                    song->win = t->win;

                    if(depth == 0)song->num_root = i;
                    else song->num_root = t->num_root;
                  

                    //start
                    bool is_magic = false;

                    /*is_magic = Get_Magic(indpl, song->player, song->player2, song->magic, song->opp_score, 
                        song->Opp, song->Snaffle, song->Bludgers, maxball, song->scoret, song->pt_magic,pt_magico,  turn, time, false);
                                    
                    if(is_magic)Apply_Magic(song->player, song->magic, song->Snaffle);*/
                  
                    song->player.angle = this->sortiesbs[i][0];
                    song->player.thrust = this->sortiesbs[i][1];

                    this->Simulate_Sorcier(song->player, 1.0, 0.75);

                   
                    ll score_s1 = 0;
                    
                                       
                    //end magic

                    for(int o = 0;o < song->Opp.size();++o){
                        /*Magic magico;
                        if(o == 0)magico = song->magic3;
                        else magico = song->magic4;

                        vector<Sim> oppo = {song->player, song->player2};
                        bool is_magico = Get_Magic(o, song->Opp[o], song->Opp[1-o], magico, song->scoret, 
                            oppo, song->Snaffle, song->Bludgers, maxball, song->opp_score, song->pt_magico,song->pt_magic,  turn, time, true );
                                        
                        

                        if(is_magico){
                            Apply_Magic(song->Opp[o], magico, song->Snaffle);
                            if(o == 0)song->magic3 = magico;
                            else song->magic4 = magico;
                        }*/

                        int mind = 10000000;
                        int ind_sn = -1;
                        for(int s = 0;s < song->Snaffle.size();++s){
                            if(song->vissn[s])continue;

                            int d = distance(song->Snaffle[s].pos, song->Opp[o].pos);
                            if(d < mind){
                                mind = d;
                                ind_sn = s;
                            }
                            
                        }

                        if(ind_sn != -1){
                            double dx = song->Snaffle[ind_sn].pos.x - song->Opp[o].pos.x;
                            double dy = song->Snaffle[ind_sn].pos.y - song->Opp[o].pos.y;
                            double angleRadians = std::atan2(dy, dx);
                            double angleDegres = angleRadians * 180.0 / M_PI;
                            if(angleDegres < 0)angleDegres += 360;
                            song->Opp[o].angle = angleDegres;
                            song->Opp[o].thrust = 150;
                            this->Simulate_Sorcier(song->Opp[o], 1.0, 0.75);
                            //this->Update_pos(song->Opp[o], 1.0);

                            if(song->Opp[o].state == 0){
                                if(mind < 400){
                                    song->Opp[o].state = 1;
                                    song->Snaffle[ind_sn].pos = song->Opp[o].pos;
                                    song->Snaffle[ind_sn].speed = song->Opp[o].speed;

                                }
                            }
                            else{
                                double dx = buto.x - song->Opp[o].pos.x;
                                double dy = buto.y - song->Opp[o].pos.y;
                                double angleRadians = std::atan2(dy, dx);
                                double angleDegres = angleRadians * 180.0 / M_PI;
                                if(angleDegres < 0)angleDegres += 360;
                                song->Snaffle[ind_sn].angle = angleDegres;
                                song->Snaffle[ind_sn].thrust = 500;
                                this->Simulate_Sorcier(song->Snaffle[ind_sn], 0.5, 0.75);
                                //this->Update_pos(song->Snaffle[ind_sn], 1.0);
                                //song->Snaffle[ind_sn].end(0.75);

                                song->Opp[o].state = 0;

                            }


                        }

                    }



                    vector<Sim> near_player = {song->player, song->player2, song->Opp[0], song->Opp[1]};
                
                    for(int s = 0;s < song->Bludgers.size();++s){
                        int minid = 2000000000;
                        int ind_plnr = -1;
                        int indp = 0;
                        for(auto &pl : near_player){
                            if(pl.id == song->Bludgers[s].state){
                                indp++;
                                continue;
                            }
                            int d = distance(pl.pos, song->Bludgers[s].pos);
                            if(d < minid){
                                minid = d;
                                ind_plnr = indp;
                            }

                            indp++;
                        }

                        if(ind_plnr == -1){
                            cerr << "ERRROR -1 " <<p.id  << " " <<  song->Bludgers[s].state<< " " << minid <<endl;
                            continue;
                        }

                        double dx = near_player[ind_plnr].pos.x - song->Bludgers[s].pos.x;
                        double dy = near_player[ind_plnr].pos.y - song->Bludgers[s].pos.y;
                        double angleRadians = std::atan2(dy, dx);
                        double angleDegres = angleRadians * 180.0 / M_PI;
                        if(angleDegres < 0)angleDegres += 360;
                        song->Bludgers[s].angle = angleDegres;
                        song->Bludgers[s].thrust = 1000;
                        this->Simulate_Sorcier(song->Bludgers[s], 8, 0.9);
                        //this->Update_pos(song->Bludgers[s], 1.0);
                        //Bludgers[s].end(0.9);
                
                    }


                    /*bool is_magic2 = false;

                    is_magic2 = Get_Magic(indpl, song->player2, song->player, song->magic2, song->opp_score, 
                        song->Opp, song->Snaffle, song->Bludgers, maxball, song->scoret, song->pt_magic,pt_magico,  turn, time, false );

                    if(is_magic2)Apply_Magic(song->player2, song->magic2, song->Snaffle);*/
                        
                    //player 2
                    
                    {
                        int mind = 10000000;
                        int ind_sn = -1;
                        for(int s = 0;s < song->Snaffle.size();++s){
                            if(song->vissn[s])continue;
                            if(indpl == 0 && song->Snaffle[s].id == song->pid_snaffle_p1 && song->Snaffle.size() >1 )continue;
                            if(indpl == 1 && song->Snaffle[s].id == song->pid_snaffle_p2 && song->Snaffle.size() > 1)continue;
                            //if(song->Snaffle.size() > 2 && song->Snaffle[s].state == 1)continue;

                            //double dc = distance(song->Opp[ind_chaseo].pos, song->Snaffle[s].pos);

                            int d = distance(song->Snaffle[s].pos, song->player2.pos);
                            if(/*(ind_def == 0 || ind_def == -1) &&*/ d < mind){
                                mind = d;
                                ind_sn = s;
                            }
                            /*else if(ind_def == 1 && dc < mind){
                                mind = dc;
                                ind_sn = s;
                            }*/
                            
                        }

                        if(ind_sn != -1){
                            
                            double dx = song->Snaffle[ind_sn].pos.x - song->player2.pos.x;
                            double dy = song->Snaffle[ind_sn].pos.y - song->player2.pos.y;
                            double angleRadians = std::atan2(dy, dx);
                            double angleDegres = angleRadians * 180.0 / M_PI;
                            if(angleDegres < 0)angleDegres += 360;
                            song->player2.angle = angleDegres;
                            song->player2.thrust = 150;
                            this->Simulate_Sorcier(song->player2, 1.0, 0.75);
                            //this->Update_pos(song->Opp[o], 1.0);

                            if(song->player2.state == 0){
                                if(mind < 400){
                                    song->player2.state = 1;
                                    song->Snaffle[ind_sn].pos = song->player2.pos;
                                    song->Snaffle[ind_sn].speed = song->player2.speed;
                                    song->Snaffle[ind_sn].state = 1;
                                }
                            }
                            else{
                                double dx = but.x - song->player2.pos.x;
                                double dy = but.y - song->player2.pos.y;
                                double angleRadians = std::atan2(dy, dx);
                                double angleDegres = angleRadians * 180.0 / M_PI;
                                if(angleDegres < 0)angleDegres += 360;
                                song->Snaffle[ind_sn].angle = angleDegres;
                                song->Snaffle[ind_sn].thrust = 500;
                                this->Simulate_Sorcier(song->Snaffle[ind_sn], 0.5, 0.75);
                                //this->Update_pos(song->Snaffle[ind_sn], 1.0);
                                //song->Snaffle[ind_sn].end(0.75);

                                song->player2.state = 0;
                                song->Snaffle[ind_sn].state = 0;

                            }
                            if(indpl == 0)song->pid_snaffle_p2 = song->Snaffle[ind_sn].id;
                            if(indpl == 1)song->pid_snaffle_p1 = song->Snaffle[ind_sn].id;


                        }

                    }

                    
                        
                    if(!is_magic){
                        
                        ind_snaffle = -1;
                        int nb_ball = this->ball_rem - song->scoret;
                        int ind_defsn = -1;
                        int ind_thsn = -1;
                        double mind1 = 2000000000;
                        int indm1 = -1;
                        for(int s = 0;s < song->Snaffle.size();++s){
                            if(song->vissn[s])continue;
                    
                            if(indpl == 0 && song->Snaffle[s].id == song->pid_snaffle_p2 && song->Snaffle.size() >1 )continue;
                            if(indpl == 1 && song->Snaffle[s].id == song->pid_snaffle_p1 && song->Snaffle.size() > 1)continue;
                          
                            //double dc = distance(song->Opp[ind_chaseo].pos, song->Snaffle[s].pos);
                                                        
                            double d = distance(song->Snaffle[s].pos, song->player.pos);
                            if(/*(ind_def == 1 || ind_def == -1)  &&*/ d < mind1){
                                mind1 = d;
                                indm1 = s;
                            }
                            /*else if(ind_def == 0 && dc < mind1){
                                mind1 = dc;
                                indm1 = s;
                            }*/

                                                            

                        }
                        if(indm1 != -1){
                            ind_snaffle = indm1;
                           
                            if(ind_snaffle != -1){
                                if(indpl == 0)song->pid_snaffle_p1 = song->Snaffle[ind_snaffle].id;
                                if(indpl == 1)song->pid_snaffle_p2 = song->Snaffle[ind_snaffle].id;
                            }
                                    
                        }

                        if(ind_snaffle != -1){
                            int dist_snaffle = distance(song->player.pos, song->Snaffle[ind_snaffle].pos);
                                    
                            score_s1 += 500000 - dist_snaffle;
                        

                            int ind_snsk = -1;
                            if(song->player.state == 0){
                                if(dist_snaffle < 400){
                                    song->player.state = 1;
                                    song->Snaffle[ind_snaffle].pos = song->player.pos;
                                    song->Snaffle[ind_snaffle].speed = song->player.speed;
                                    song->Snaffle[ind_snaffle].state = 1;
                                }
                                ind_snsk = ind_snaffle;
                            }
                            else{
                                if(ind_snaffle != -1){
                                    vector<Sim> OPPO;
                                    OPPO.insert(OPPO.end(), song->Opp.begin(), song->Opp.end());
                                    OPPO.insert(OPPO.end(), song->Bludgers.begin(), song->Bludgers.end());
                                    Vector2 tir = bestShot(song->player.pos, OPPO, but);
                                    double dx = tir.x - song->player.pos.x;
                                    double dy = tir.y - song->player.pos.y;
                                    double angleRadians = std::atan2(dy, dx);
                                    double angleDegres = angleRadians * 180.0 / M_PI;
                                    if(angleDegres < 0)angleDegres += 360;
                                    song->Snaffle[ind_snaffle].angle = angleDegres;
                                    song->Snaffle[ind_snaffle].thrust = 500;
                                    this->Simulate_Sorcier(song->Snaffle[ind_snaffle], 0.5, 0.75);
                                   
                                    ind_snsk = ind_snaffle;

                                    song->Snaffle[ind_snaffle].state = 0;

                                    song->player.state = 0;

                                }

                            }

                        }

                    }

                    for(int s = 0;s < song->Snaffle.size();++s){
                        if(song->vissn[s])continue;

                        Vector2 contact = getBorderIntersection(song->Snaffle[s].pos, song->Snaffle[s].speed);
                        Sim sc;
                        sc.pos = contact;
                        sc.speed = {0.0f,0.0f};
                        sc.radius = 400.0f;
                        double t = this->CollisionTime(song->Snaffle[s], sc);
                    
                        if(t >= 0.0f && t<= 1.0f){
                            this->Update_pos(song->Snaffle[s], t);
                            
    

                        }
                        else{
                            this->Update_pos(song->Snaffle[s], 1.0);
                        }

                        if(song->Snaffle[s].pos.x <75)song->Snaffle[s].bounce(HORIZONTAL);
                        if(song->Snaffle[s].pos.x >15925)song->Snaffle[s].bounce(HORIZONTAL);
                        if(song->Snaffle[s].pos.y <75)song->Snaffle[s].bounce(VERTICAL);
                        if(song->Snaffle[s].pos.y >7425)song->Snaffle[s].bounce(VERTICAL);

                        if ((contact.x  == 0 || contact.x == 16000) && (contact.y >= 2025 && contact.y <= 5415)){
                            if(song->player.idteam == 0 && song->Snaffle[s].pos.x < 0){
                                song->vissn[s] = true;
                                score_s1-= 1000;
                                song->opp_score++;
                                if(song->opp_score == ball_rem){
                                    score_s1 -= 10000000;
                                    song->terminal = true;
                                    song->win = 0;

                                }
                            }
                            if(song->player.idteam == 1 && song->Snaffle[s].pos.x > 16000){
                                song->vissn[s] = true;
                                score_s1 -= 1000;
                                song->opp_score++;
                                if(song->opp_score == ball_rem){
                                    score_s1 -= 10000000;
                                    song->terminal = true;
                                    song->win = 0;
                                }
                            }

                            if(song->player.idteam == 0 && song->Snaffle[s].pos.x > 16000){
                                song->vissn[s] = true;
                                score_s1 += 1000000;
                                song->scoret++;
                                if(song->scoret == ball_rem){
                                    score_s1 += 10000000;
                                    song->terminal = true;
                                    song->win = 1;
                                }
                            }
                            if(song->player.idteam == 1 && song->Snaffle[s].pos.x < 0){
                                song->vissn[s] = true;
                                score_s1 += 1000000;
                                song->scoret++;
                                if(song->scoret == ball_rem){
                                    score_s1 += 10000000;
                                    song->terminal = true;
                                    song->win = 1;
                                }
                            }

                        }

                        


                    }

                    
                    
                    //end
                    vector<bool> col_blg(song->Bludgers.size(), false);
                    vector<bool> col_opp(2, false);



                    bool col_bl = false, col_bl2 = false;
                    int minb = 1000000;
                    int ind_b = -1;
                    for(int j = 0;j < song->Bludgers.size();++j){
                        int d = distance(song->player.pos, song->Bludgers[j].pos);
                        if(d < minb){
                            minb = d;
                        }

                        int d1 = distance(song->player2.pos, song->Bludgers[j].pos);
                        int d2 = distance(song->Opp[0].pos, song->Bludgers[j].pos);
                        int d3 = distance(song->Opp[1].pos, song->Bludgers[j].pos);

                        vector<pair<int, int>> vd = {{d,0}, {d1,1},{d2,2},{d3,3}};
                        sort(vd.begin(), vd.end());

                        Sim s;
                        float t = 0;
                        if(vd[0].second == 0)s =song->player;
                        if(vd[0].second == 1)s =song->player2;
                        if(vd[0].second == 2)s =song->Opp[0];
                        if(vd[0].second == 3)s =song->Opp[1];

                        t = this->CollisionTime(s, song->Bludgers[j]);
                                                                    
                        if(t >= 0.0f && t<= 1.0f){
                            this->Update_pos(song->Bludgers[j], t);
                            if(vd[0].second == 0){
                                this->Update_pos(song->player, t);
                                song->player.bounce(song->Bludgers[j]);
                                song->Bludgers[j].state = song->player.id;
                                col_bl = true;
                            }
                            else if(vd[0].second == 1){
                                this->Update_pos(song->player2, t);
                                song->player2.bounce(song->Bludgers[j]);
                                song->Bludgers[j].state = song->player2.id;
                                col_bl2 = true;
                            }
                            else if(vd[0].second == 2){
                                this->Update_pos(song->Opp[0], t);
                                song->Opp[0].bounce(song->Bludgers[j]);
                                song->Bludgers[j].state = song->Opp[0].id;
                                col_opp[0] = true;
                            }
                            else if(vd[0].second == 3){
                                this->Update_pos(song->Opp[1], t);
                                song->Opp[1].bounce(song->Bludgers[j]);
                                song->Bludgers[j].state = song->Opp[1].id;
                                col_opp[1] = true;
                            }

                            col_blg[j] = true;
                                                        
                            
                            

                        }

                    }

                    
                    for(int j = 0;j < song->Opp.size();++j){
                        float t = this->CollisionTime(song->player, song->Opp[j]);
                        if(t >= 0.0f && t<= 1.0f){
                            this->Update_pos(song->player, t);
                            this->Update_pos(song->Opp[j], t);
                            song->player.bounce(song->Opp[j]);
                            col_bl = true;
                            col_opp[j] = true;

                        }

                    }

                    for(int j = 0;j < song->Opp.size();++j){
                        float t = this->CollisionTime(song->player2, song->Opp[j]);
                        if(t >= 0.0f && t<= 1.0f){
                            this->Update_pos(song->player2, t);
                            this->Update_pos(song->Opp[j], t);
                            song->player2.bounce(song->Opp[j]);
                            col_bl2 = true;
                            col_opp[j] = true;

                        }

                    }

                    float tc = this->CollisionTime(song->player, song->player2);
                    if(tc >= 0.0f && tc<= 1.0f){
                        this->Update_pos(song->player, tc);
                        this->Update_pos(song->player2, tc);
                        song->player.bounce(song->player2);
                        col_bl = true;
                        col_bl2 = true;

                    }
                    
                    if(minb > 650)
                        score_s1 += 100000;

                                        
                    if(!col_bl)this->Update_pos(song->player, 1.0);
                    if(!col_bl2)this->Update_pos(song->player2, 1.0);

                    for(int j = 0;j < 2;++j){
                        if(!col_opp[j])this->Update_pos(song->Opp[j], 1.0);
                    }

                    for(int j = 0;j < song->Bludgers.size();++j){
                        if(!col_blg[j])this->Update_pos(song->Bludgers[j], 1.0);
                    }
                
                    if(song->player.pos.x <400)song->player.bounce(HORIZONTAL);
                    if(song->player.pos.x >15600)song->player.bounce(HORIZONTAL);
                    if(song->player.pos.y <400)song->player.bounce(VERTICAL);
                    if(song->player.pos.y >7100)song->player.bounce(VERTICAL);

                    if(song->player.pos.x <400)song->player.pos.x = 400;
                    if(song->player.pos.x >15600)song->player.pos.x = 15600;
                    if(song->player.pos.y <400)song->player.pos.y = 400;
                    if(song->player.pos.y >7100)song->player.pos.y = 7100; 
                    
                    if(song->player2.pos.x <400)song->player2.bounce(HORIZONTAL);
                    if(song->player2.pos.x >15600)song->player2.bounce(HORIZONTAL);
                    if(song->player2.pos.y <400)song->player2.bounce(VERTICAL);
                    if(song->player2.pos.y >7100)song->player2.bounce(VERTICAL);

                    if(song->player2.pos.x <400)song->player2.pos.x = 400;
                    if(song->player2.pos.x >15600)song->player2.pos.x = 15600;
                    if(song->player2.pos.y <400)song->player2.pos.y = 400;
                    if(song->player2.pos.y >7100)song->player2.pos.y = 7100;
                    
                    for(int j = 0;j < 2;++j){
                        if(song->Opp[j].pos.x <400)song->Opp[j].bounce(HORIZONTAL);
                        if(song->Opp[j].pos.x >15600)song->Opp[j].bounce(HORIZONTAL);
                        if(song->Opp[j].pos.y <400)song->Opp[j].bounce(VERTICAL);
                        if(song->Opp[j].pos.y >7100)song->Opp[j].bounce(VERTICAL);

                        if(song->Opp[j].pos.x <400)song->Opp[j].pos.x = 400;
                        if(song->Opp[j].pos.x >15600)song->Opp[j].pos.x = 15600;
                        if(song->Opp[j].pos.y <400)song->Opp[j].pos.y = 400;
                        if(song->Opp[j].pos.y >7100)song->Opp[j].pos.y = 7100; 
                    }

                    for(int s = 0;s < song->Bludgers.size();++s)song->Bludgers[s].end(0.9);
                    for(int s = 0;s < song->Snaffle.size();++s){
                        if(song->vissn[s])continue;
                        song->Snaffle[s].end(0.75);

                    }
                    for(int s = 0;s < song->Opp.size();++s)song->Opp[s].end(0.75);

                    song->player2.end(0.75);
                    song->player.end(0.75);

                    if(song->pt_magic < 100)
                        song->pt_magic++;


                    
                    song->score = t->score + score_s1;

                   
                    if(!song->terminal)
                        TF.insert(song);
                    else{
                    
                        if(song->score > scoref ||song->win == 1){
                            T0 = *song;
                            scoref = song->score;
                        }

                        /*if(song->win == 1){
                            end = true;

                        }*/

                    }

                    //if(end)break;

                }

                //if(end)break;

            }

            //if(end)break;

            if(TF.empty())break;
            beam.swap(TF);
            if((*beam.begin())->score > scoref){
                T0 = *(*beam.begin());
                scoref = (*beam.begin())->score;
            }

            //cerr << depth << ",  time="<< maxt << endl;

            depth++;


        }

        cerr << depth << ",  time="<< maxt << endl;
        cerr << "SCORE=" << T0.score << endl;


        int num_s1 = T0.num_root;
        cerr << num_s1 << endl;
        int angle = this->sortiesbs[num_s1][0];
        int thrust = this->sortiesbs[num_s1][1];

        Vector2 dir;
        int x, y;
     
        dir.x = cosAngles[(int)angle] * 10000.0 ;
        dir.y = sinAngles[(int)angle] * 10000.0 ;

        x = p.pos.x + dir.x;
        y = p.pos.y + dir.y;

        string ans = "MOVE " + to_string(x) + " " + to_string(y) + " " + to_string(thrust);

        return ans;

    }


};




/**
 * Grab Snaffles and try to throw them through the opponent's goal!
 * Move towards a Snaffle and use your team id to determine where you need to throw it.
 **/

int main()
{
    fast_srand(42);

    vector<vector<Vector2>>dcell;

    for(int i = 0;i <= 7500;i+=200){
        vector<Vector2> dc;
        for(int j = 0;j <= 16000;j+= 200){
            dc.push_back(Vector2{(float)j, (float)i});
        }
        dcell.push_back(dc);
    }

    
    
    for (int i = 0; i < 360; ++i) {
        cosAngles[i] = cos((double)i * TO_RAD);
        sinAngles[i] = sin((double)i* TO_RAD);
    }

    int my_team_id; // if 0 you need to score on the right of the map, if 1 you need to score on the left
    cin >> my_team_id; cin.ignore();
    Simulation simul = Simulation(3, 2);
    simul.dcell = dcell;

    //---------
    Vector2 but;
  
    int cage = -1;
    if(my_team_id == 0){
        but = {16500, 3750};
        double mind = 2000000.0;
        for(int i = 0;i < dcell.size();++i){
            for(int j = 0;j < dcell[0].size();++j){
                double d = distance(dcell[i][j], but);
                if(d < mind){
                    mind = d;
                    VEND1 ={(float)j, (float)i};
                }
            }
        }
        
    }
    else{
        but = {-500, 3750};
        double mind = 2000000.0;
        for(int i = 0;i < dcell.size();++i){
            for(int j = 0;j < dcell[0].size();++j){
                double d = distance(dcell[i][j], but);
                if(d < mind){
                    mind = d;
                    VEND1 ={(float)j, (float)i};
                }
            }
        }
        
    }


    int turn = 0;
    int maxball = 0;
    // game loop
    while (1) {
        int my_score;
        int my_magic;
        cin >> my_score >> my_magic; cin.ignore();
        int opponent_score;
        int opponent_magic;
        cin >> opponent_score >> opponent_magic; cin.ignore();
        int entities; // number of entities still in game
        cin >> entities; cin.ignore();

        vector<Sim> wizard, opp_wizard, snaffle, bludgers;
        for (int i = 0; i < entities; i++) {
            int entity_id; // entity identifier
            string entity_type; // "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" (or "BLUDGER" after first league)
            int x; // position
            int y; // position
            int vx; // velocity
            int vy; // velocity
            int state; // 1 if the wizard is holding a Snaffle, 0 otherwise
            cin >> entity_id >> entity_type >> x >> y >> vx >> vy >> state; cin.ignore();

            if(entity_type == "WIZARD"){
                Sim sm;
                sm.pos = {x, y};
                sm.speed = {vx, vy};
                sm.state = state;
                sm.id = entity_id;
                sm.idteam = my_team_id;
                sm.radius = 400;
                sm.m = 1.0;
                wizard.push_back(sm);


            }
            else if(entity_type == "OPPONENT_WIZARD"){
                Sim sm;
                sm.pos = {x, y};
                sm.speed = {vx, vy};
                sm.state = state;
                sm.id = entity_id;
                sm.idteam = 1-my_team_id;
                sm.radius = 400;
                sm.m = 1.0;

                opp_wizard.push_back(sm);


            }
            else if(entity_type == "SNAFFLE"){
                Sim sm;
                sm.id = entity_id;
                sm.pos = {x, y};
                sm.speed = {vx, vy};
                sm.state = state;
                sm.radius = 150;
                sm.m = 0.5;
                snaffle.push_back(sm);


            }
            else if(entity_type == "BLUDGER"){
                Sim sm;
                sm.pos = {x, y};
                sm.speed = {vx, vy};
                sm.state = state;
                sm.id = entity_id;
                sm.idteam = my_team_id;
                sm.radius = 200;
                sm.m = 8;

                bludgers.push_back(sm);


            }

        }

        if(turn == 0){
            maxball = snaffle.size();
            simul.ball_rem = snaffle.size() / 2 + 1;
        }
        

        int time = 98;
        if(turn == 0)time = 998;

        

        auto startm = high_resolution_clock::now();;
             
        cout << simul.BeamSearch(0, wizard[0], wizard[1], opp_wizard, snaffle, bludgers, maxball, my_score, my_magic,opponent_magic, turn, time/2, opponent_score) << endl;
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - startm);
        cout << simul.BeamSearch(1, wizard[1], wizard[0], opp_wizard, snaffle, bludgers, maxball, my_score, my_magic,opponent_magic, turn, time-duration.count(), opponent_score) << endl;
        
        //for(int i = 0;i< 2;++i)cout << ans[i] << endl;


        ++turn;
        
    }
}