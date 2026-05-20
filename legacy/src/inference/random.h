#pragma once

#include <random>

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
	std::vector<T> values;
	for (T value = start; value < stop; value += step)
		values.push_back(value);
	return values;
}

template<typename T>
std::vector<T> arange(T stop) {
	return arange(0, stop, 1);
}

int random_on_range(int n, std::mt19937 *engine) {
  std::uniform_int_distribution<int> dist(0,n-1);
  return dist(*engine);
}

double random_on_unit(std::mt19937 *engine) {
  std::uniform_real_distribution<double> dist(0.,1.);
  return dist(*engine);
}

double random_on_range(double mi, double ma, std::mt19937 *engine) {
  std::uniform_real_distribution<double> dist(mi,ma);
  return dist(*engine);
}

int random_on_range(int mi, int ma, std::mt19937 *engine) {
  if (mi > ma) {
    std::ostringstream buffer;
    buffer << "random_on_range: min=" << mi << " > max=" << ma;
    throw std::invalid_argument(buffer.str());
  }
  if (mi == ma) {
    return mi;
  }
  std::uniform_int_distribution<int> dist(mi,ma);
  return dist(*engine);
}

template<typename T>
T random_element(std::vector<T> &items, std::mt19937 *engine) {
  int index = random_on_range(items.size(), engine);
  return items[index];
}

int random_element_int(std::vector<int> items, std::mt19937 *engine) {
  int index = random_on_range(items.size(), engine);
  return items[index];
}

template<typename T>
std::vector<T> random_subset(std::vector<T> &items, std::mt19937 *engine) {
  int n = random_on_range(items.size(), engine) + 1;
  std::vector<int> idx = arange((int)items.size());
  std::shuffle(idx.begin(), idx.end(), *engine);
  std::vector<T> output;
  for (int i=0; i<n; i++) {
    output.push_back( items[idx[i]] );
  }
  return output;
}