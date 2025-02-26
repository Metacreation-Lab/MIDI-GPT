#include <array>
#include <vector>
#include <iostream>
#include <random>

template <typename T>
using vmatrix = std::vector<std::vector<T>>;

template <typename T>
class cmatrix {
public:
    cmatrix() {
        N = 0;
        M = 0;
        m_data = vmatrix<T>(0, std::vector<T>(0, 0));
    }
    cmatrix(int n, int m, int value) {
        N = n;
        M = m;
        m_data = vmatrix<T>(n, std::vector<T>(m, value));
    }
    cmatrix(const cmatrix<T> &x) {
        N = x.N;
        M = x.M;
        m_data = vmatrix<T>(N, std::vector<T>(M, 0));
        for (int i=0; i<N; i++) {
            for (int j=0; j<M; j++) {
                m_data[i][j] = x.m_data[i][j];
            }
        }
    }
    inline cmatrix<T> & operator=(const cmatrix<T> &x) {
        N = x.N;
        M = x.M;
        m_data = vmatrix<T>(N, std::vector<T>(M, 0));
        for (int i=0; i<N; i++) {
            for (int j=0; j<M; j++) {
                m_data[i][j] = x.m_data[i][j];
            }
        }
        return *this;
    }
    cmatrix(const std::vector<std::vector<T>> &x) {
        assert(x.size() > 0);
        N = x.size();
        M = x[0].size();
        m_data = vmatrix<T>(N, std::vector<T>(M, 0));
        for (int i=0; i<N; i++) {
            assert((int)x[i].size() == M);
            for (int j=0; j<M; j++) {
                m_data[i][j] = x[i][j];
            }
        }
    }
    bool same_shape(cmatrix<T> &x) {
        return (N == x.N) && (M == x.M);
    }
    bool is_shape(int n, int m) {
        return (N == n) && (M == m);
    }
    cmatrix<T> transpose() {
        cmatrix<T> c(M, N, 0);
        for (int i=0; i<N; i++) {
            for (int j=0; j<M; j++) {
                c.m_data[j][i] = m_data[i][j];
            }
        }
        return c;
    }

    int N;
    int M;
    vmatrix<T> m_data;
};


template <typename T>
vmatrix<T> random_boolean_matrix(int n, int m, double p, std::mt19937 *e) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    vmatrix<T> x(n, std::vector<T>(m, 0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            x[i][j] = (dist(*e) < p);
        }
    }
    return x;
}

template <typename T>
vmatrix<T> ones(int n, int m) {
    return vmatrix<T>(n, std::vector<T>(m, 1));
}

template <typename T>
vmatrix<T> zeros(int n, int m) {
    return vmatrix<T>(n, std::vector<T>(m, 0));
}

template <typename T, typename F>
vmatrix<T> binary_op(const vmatrix<T> &a, const vmatrix<T> &b, F &&func) {
    assert(a.size() == b.size());
    vmatrix<T> c(a.size(), std::vector<T>(a[0].size(), 0));
    for (int i=0; i<(int)a.size(); i++) {
        assert(a[i].size() == b[i].size());
        for (int j=0; j<(int)a[i].size(); j++) {
            c[i][j] = func(a[i][j], b[i][j]);
        }
    }
    return c;
}

template <typename T>
vmatrix<T> operator& (const vmatrix<T> &a, const vmatrix<T> &b) {
    return binary_op(a, b, [](T x, T y) { return x & y; });
}

template <typename T>
vmatrix<T> operator| (const vmatrix<T> &a, const vmatrix<T> &b) {
    return binary_op(a, b, [](T x, T y) { return x | y; });
}


template <typename T>
bool all(const vmatrix<T> &x) {
    for (const auto &row : x) {
        for (const auto &elem : row) {
            if (!elem) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool all(const cmatrix<T> &x) {
    return all(x.m_data);
}

template <typename T>
bool any(const std::vector<T> &x) {
    for (const auto &elem : x) {
        if (elem) {
            return true;
        }
    }
    return false;
}

template <typename T>
bool any(const vmatrix<T> &x) {
    for (const auto &row : x) {
        for (const auto &elem : row) {
            if (elem) {
                return true;
            }
        }
    }
    return false;
}

template <typename T>
bool any(const cmatrix<T> &x) {
    return any(x.m_data);
}

template <typename T>
int sum(const vmatrix<T> &x) {
    int total = 0;
    for (const auto &row : x) {
        for (const auto &elem : row) {
            total += (int)elem;
        }
    }
    return total;
}

int sum(const cmatrix<bool> &x) {
    return sum(x.m_data);
}

template <typename T>
bool equal(const vmatrix<T> &a, const vmatrix<T> &b) {
    if(a.size() != b.size()) {
        return false;
    }
    for (int i=0; i<(int)a.size(); i++) {
        if(a[i].size() != b[i].size()) {
            return false;
        }
        for (int j=0; j<(int)a[i].size(); j++) {
            if (a[i][j] != b[i][j]) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool operator==(const vmatrix<T> &a, const vmatrix<T> &b) {
    return equal(a, b);
}

template <typename T>
void show(const vmatrix<T> &x) {
    for (const auto &row : x) {
        for (const auto &elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void show(const cmatrix<T> &x) {
    show(x.m_data);
}

template <typename T>
void show(const std::vector<T> &x) {
    for (const auto &elem : x) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

template <typename T>
T clamp(T x, T min, T max) {
    return std::max(std::min(x, max), min);
}

template <typename T, typename F>
cmatrix<T> unary_op(cmatrix<T> a, F &&func) {
    cmatrix<T> c(a.N, a.M, 0);
    for (int i=0; i<a.N; i++) {
        for (int j=0; j<a.M; j++) {
            c.m_data[i][j] = func(a.m_data[i][j]);
        }
    }
    return c;
}

template <typename T, typename F>
cmatrix<T> binary_op(cmatrix<T> a, cmatrix<T> b, F &&func) {
    assert(a.same_shape(b));
    cmatrix<T> c(a.N, a.M, 0);
    for (int i=0; i<a.N; i++) {
        for (int j=0; j<a.M; j++) {
            c.m_data[i][j] = func(a.m_data[i][j], b.m_data[i][j]);
        }
    }
    return c;
}

template <typename T>
cmatrix<T> operator~(cmatrix<T> a) {
    return unary_op<T>(a, [](T x) { return !x; });
}

template <typename T>
cmatrix<T> operator&(cmatrix<T> a, cmatrix<T> b) {
    return binary_op<T>(a, b, [](T x, T y) { return x & y; });
}

template <typename T>
cmatrix<T> operator*(cmatrix<T> a, cmatrix<T> b) {
    return binary_op<T>(a, b, [](T x, T y) { return x * y; });
}

template <typename T>
cmatrix<T> operator|(cmatrix<T> a, cmatrix<T> b) {
    return binary_op<T>(a, b, [](T x, T y) { return x | y; });
}

// operation along axis
template <typename T, typename F>
cmatrix<T> op_axis(cmatrix<T> a, int axis, F &&func) {
    assert(axis == 0 || axis == 1);
    auto x(a);
    if (axis == 0) {
        x = x.transpose();
    }
    cmatrix<T> y(x.N, x.M, 0);
    for (int i=0; i<x.N; i++) {
        for (int j=0; j<x.M; j++) {
            y.m_data[i][j] = func(x.m_data[i]);
        }
    }
    if (axis == 0) {
        y = y.transpose();
    }
    return y;
}

template <typename T>
cmatrix<T> max_along_axis(cmatrix<T> a, int axis) {
    return op_axis(a, axis, [](const std::vector<T>& x) { return *std::max_element(x.begin(), x.end()); });
}

template <typename T>
cmatrix<T> getrange(cmatrix<T> a, int is, int ie, int js, int je) {
    assert(a.N >= ie && a.M >= je);
    cmatrix<T> b(ie-is, je-js, 0);
    for (int i=is; i<ie; i++) {
        for (int j=js; j<je; j++) {
            b.m_data[i-is][j-js] = a.m_data[i][j];
        }
    }
    return b;
}

template <typename T>
void setrange(cmatrix<T> &x, cmatrix<T> y, int is, int ie, int js, int je) {
    assert(x.N >= ie && x.M >= je);
    assert(y.is_shape(ie-is, je-js));
    for (int i=is; i<ie; i++) {
        for (int j=js; j<je; j++) {
            x.m_data[i][j] = y.m_data[i-is][j-js];
        }
    }
}

template <typename T>
void setrange(cmatrix<T> &x, T y, int is, int ie, int js, int je) {
    for (int i=is; i<ie; i++) {
        for (int j=js; j<je; j++) {
            x.m_data[i][j] = y;
        }
    }
}

template <typename T>
cmatrix<T> vector_to_matrix(std::vector<T> x, int M) {
    cmatrix<T> y(x.size(), M, 0);
    for (int i=0; i<(int)x.size(); i++) {
        for (int j=0; j<M; j++) {
            y.m_data[i][j] = x[i];
        }
    }
    return y;
}

class STEP {
public:
    STEP (int sstart, int eend, vmatrix<bool> &sstep, vmatrix<bool> &ccontext) {
        start = sstart;
        end = eend;
        step = sstep;
        context = ccontext;
        initialize();
    }

    STEP (const STEP &old) {
        start = old.start;
        end = old.end;
        step = old.step;
        context = old.context;
        initialize();
    }

    STEP () {
        start = 0;
        end = 0;
        throw std::runtime_error("STEP constructor called with no arguments");
    }

    virtual ~STEP () {}

    void initialize() {
        int track_count = 0;
        int num_tracks = step.size();
        std::set<int> track_set;
        for (int i=0; i<num_tracks; i++) {
            bool track_used = false;
            for (int j=start; j<end; j++) {
                if (step[i][j]) {
                    bars_to_generate.insert( std::make_tuple(track_count,j-start) );
                    bar_mapping.push_back( std::make_tuple(track_count,j-start,i,j) );
                }
                if (step[i][j] || context[i][j]) {
                    track_set.insert( i );
                    track_used = true;
                }
            }
            if (track_used) {
                track_count++;
            }
        }
        tracks = std::vector<int>(track_set.begin(), track_set.end());
        assert(bars_to_generate.size() > 0);
    }

    std::vector<int> get_tracks() const {
        return tracks;
    }

    std::set<std::tuple<int,int>> get_bars_to_generate() const{
        return bars_to_generate;
    }

    std::vector<std::tuple<int,int,int,int>> get_bar_mapping() const {
        return bar_mapping;
    }

    int generated_bar_count() const {
        return sum(step);
    }

    int start;
    int end;
    vmatrix<bool> step;
    vmatrix<bool> context;

private:
    std::set<std::tuple<int,int>> bars_to_generate;
    std::vector<std::tuple<int,int,int,int>> bar_mapping;
    std::vector<int> tracks;
};

class HyperParam {
public:
    HyperParam () {
        _model_dim = 4;
        _tracks_per_step = 1;
        _bars_per_step = 4;
        _shuffle = false;
        _percentage = 100;
    }
    int model_dim() const {
        return _model_dim;
    }
    int tracks_per_step() const {
        return _tracks_per_step;
    }
    int bars_per_step() const {
        return _bars_per_step;
    }
    bool shuffle() const {
        return _shuffle;
    }
    int percentage() const {
        return _percentage;
    }
    int _model_dim;
    int _tracks_per_step;
    int _bars_per_step;
    bool _shuffle;
    int _percentage;
};

void find_steps_inner(std::vector<STEP> &steps, cmatrix<bool> &selection_matrix, cmatrix<bool> &resample_mask, cmatrix<bool> &ignore_mask, bool autoregressive, cmatrix<bool> &generated, midi::HyperParam *param) {

    int model_dim = param->model_dim();
    int tracks_per_step = clamp(param->tracks_per_step(), 1, selection_matrix.N);
    int bars_per_step = clamp(param->bars_per_step(), 1, model_dim);
    int current_num_steps = steps.size();
    int num_context = autoregressive ? model_dim - bars_per_step : (model_dim - bars_per_step) / 2;
    int nt = selection_matrix.N;
    int nb = selection_matrix.M;

    auto sel(selection_matrix);
    auto covered = cmatrix<bool>(nt,nb,0);
    auto tracks_to_consider = arange(0, nt, 1);

    sel = autoregressive ? sel & resample_mask : sel & ~resample_mask;

    std::vector<std::tuple<int,int>> ijs;
    for (int i=0; i<(int)sel.N; i=i+tracks_per_step) {
        for (int j=0; j<(int)sel.M; j=j+bars_per_step) {
            ijs.push_back( std::make_tuple(i,j) );
        }
    }

    for (const auto &ij : ijs) {
        int i = std::get<0>(ij);
        int j = std::get<1>(ij);
        int num_tracks = std::min(tracks_per_step,(int)sel.N-i);
        auto kernel = cmatrix<bool>(num_tracks,model_dim,0);
        auto step = cmatrix<bool>(nt,nb,0);
        auto context = cmatrix<bool>(nt,nb,0);

        int t = 0;
        if (autoregressive) {
            // for the first step we have no generated material to 
            // condition on so we use entire model window
            // after the first step (j>0) we only generate bars_per_step bars
            int right_offset = std::max((j + model_dim) - nb,0);
            t = std::min(j, nb - model_dim);
            setrange(kernel, true, 0, num_tracks, (j>0)*(num_context+right_offset), model_dim);
        }
        else {
            // we want to have the generated bars at the center
            // this is not possible at beginning and end so we adjust for those cases
            t = clamp(j - num_context, 0, nb - model_dim);
            setrange(kernel, true, 0, num_tracks, j-t, j-t+bars_per_step);
        }

        int a = i + num_tracks;
        int b = t + model_dim;
        setrange(step, getrange(sel, i, a, t, b) * kernel, i, a, t, b);
        if (autoregressive) {
            setrange(step, getrange(step, i, a, t, b) & ~getrange(generated, i, a, t, b), i, a, t, b);
        }
        setrange(context, ~getrange(ignore_mask, 0, nt, t, b) & ~getrange(step, 0, nt, t, b), 0, nt, t, b);
        if (autoregressive) {
            auto h = max_along_axis(sel, 1);
            setrange(context, getrange((h * generated) | (~h * context), 0, nt, t, b), 0, nt, t, b);
        }

        if (any(step)) {
            steps.push_back(STEP(t, t+model_dim, step.m_data, context.m_data));
        }

        setrange(generated, getrange(generated, i, a, t, b) | getrange(step, i, a, t, b), i, a, t, b);
        setrange(covered, getrange(covered, i, a, t, b) | kernel, i, a, t, b);

    }

    if (!all(covered)) {
        throw std::runtime_error("PIECE IS ONLY PARTIALLY COVERED");
    }

    if ((!autoregressive) && (param->shuffle())) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(steps.begin() + current_num_steps, steps.end(), g);
    }
    if ((!autoregressive) && (param->percentage() < 100) && ((int)steps.size() > current_num_steps)) {
        int non_autoreg_steps = steps.size() - current_num_steps;
        int new_size = non_autoreg_steps * ((float)param->percentage() / 100.);
        steps.resize(current_num_steps + std::max(new_size,1));
    }

}