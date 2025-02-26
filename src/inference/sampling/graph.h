#pragma once

#include <map>
#include <set>
#include <vector>
#include <tuple>
#include <sstream>

namespace sampling {

// define printing methods
std::ostream& operator<<(std::ostream& os, const std::tuple<midi::TOKEN_TYPE,int> &obj) {
    return os << "(" << util_protobuf::enum_to_string(std::get<0>(obj)) << "," << std::get<1>(obj) << ")";
}

std::string toString(const std::tuple<midi::TOKEN_TYPE,int> &obj) {
	return std::string("(") + util_protobuf::enum_to_string(std::get<0>(obj)) + "," + std::to_string(std::get<1>(obj)) + ")";
}

template <typename T>
class DIGRAPH_NODE {
public:
	DIGRAPH_NODE(const T &x) {
		node_id = x;
	}
	std::set<T> edges;
	std::set<T> in_edges;
	T node_id;
};

template <typename T>
class DIGRAPH {
public:
	DIGRAPH() {
		traversal_started = false;
	}
	DIGRAPH(const std::vector<std::vector<T>> &paths) {
		traversal_started = false;
		build_from_paths(paths);
	}
	void remove_edges_to_node(const T &v) {
		data_structures::LOGGER(data_structures::to_str("REMOVING EDGES TO NODE ", toString(v)));
		for (auto kv : nodes) {
			nodes.find(kv.first)->second.edges.erase(v);
			nodes.find(kv.first)->second.in_edges.erase(v);
		}
	}
	void remove_node(const T &v) {
		data_structures::LOGGER(data_structures::to_str("REMOVING NODE ", toString(v)));
		auto node = nodes.find(v);
		if (node != nodes.end()) {
			std::set<T> out_edges = node->second.edges;
			for (const auto &pre : node->second.in_edges) {
				for (const auto &e : out_edges) {
					nodes.find(pre)->second.edges.insert( e );
				}
				nodes.find(pre)->second.edges.erase( v );
			}
			std::set<T> in_edges = node->second.in_edges;
			for (const auto &post : node->second.edges) {
				for (const auto &e : in_edges) {
					nodes.find(post)->second.in_edges.insert( e );
				}
				nodes.find(post)->second.in_edges.erase( v );
			}
			remove_edges_to_node(v);
			nodes.erase(v);
		}
	}
	void remove_nodes(const std::vector<T> &vs) {
		for (const auto &v : vs) {
			remove_node(v);
		}
	}
	void remove_nodes_wo_connecting(const std::vector<T> &vs) {
		for (const auto &v : vs) {
			remove_edges_to_node(v);
			nodes.erase(v);
		}
	}
	void add_node(const T &v) {
		if (nodes.find(v) == nodes.end()) {
			data_structures::LOGGER(data_structures::to_str("ADDING NODE ", toString(v)));
			nodes.insert( std::make_pair(v,DIGRAPH_NODE<T>(v)) );
		}
	}
	void add_edge(const T &u, const T &v) {
		data_structures::LOGGER(data_structures::to_str("ADDING EDGE ", toString(u), " -> ", toString(v)));
		add_node(u);
		add_node(v);
		nodes.find(u)->second.edges.insert(v);
		nodes.find(v)->second.in_edges.insert(u);
	}
	void add_path(const std::vector<T> &path) {
		for (int i=0; i<(int)path.size()-1; i++) {
			add_edge(path[i], path[i+1]);
		}
	}
	void build_from_paths(const std::vector<std::vector<T>> &paths) {
		for (const auto &path : paths) {
			add_path(path);
		}
	}
	bool check_path(const T &u, const T &v, int depth, int max_depth) {
		if (depth < max_depth) {
			auto choices = get_next_nodes(u);
			for (const auto &e : choices) {
				if (e == v) {
					return true;
				}
			}
			for (const auto &e : choices) {
				if (check_path(e, v, depth + 1, max_depth)) {
					return true;
				}
			}
		}
		return false;
	}
	std::vector<T> get_previous_nodes(const T &v) {
		auto it = nodes.find(v);
		if (it == nodes.end()) {
			std::ostringstream buffer;
			buffer << "ERROR : INVALID NODE IN DIGRAPH (" << v << ")";
			throw std::runtime_error(buffer.str());
		}
		std::vector<T> previous_tokens;
		for (const auto &e : it->second.in_edges) {
			previous_tokens.push_back(e);
		}
		return previous_tokens;
	}
	std::vector<T> get_next_nodes(const T &v) {
		auto it = nodes.find(v);
		if (it == nodes.end()) {
			std::ostringstream buffer;
			buffer << "ERROR [get_next_nodes()] : INVALID NODE IN DIGRAPH (" << v << ")";
			throw std::runtime_error(buffer.str());
		}
		std::vector<T> next_tokens;
		for (const auto &e : it->second.edges) {
			next_tokens.push_back(e);
		}
		return next_tokens;
	}
	T infer_node(const int &last_token, encoder::ENCODER *enc) {
		data_structures::LOGGER(data_structures::to_str("INFERRING NODE FROM TOKEN ", (enc->rep->pretty(last_token))));
		data_structures::LOGGER(data_structures::to_str("CURRENT NODE ", toString(current_node)));
		std::vector<T> next_nodes = get_next_nodes(current_node);
		midi::TOKEN_TYPE tt = enc->rep->get_token_type(last_token);
		if (tt == midi::TOKEN_PIECE_START) {
			return std::make_tuple(midi::TOKEN_PIECE_START,0); // special case at the start of the token sequence
		}
		if (next_nodes.size() == 0) {
			std::ostringstream buffer;
			buffer << "ERROR : NO NEXT TOKENS IN DIGRAPH (" << current_node << ")";
			throw std::runtime_error(buffer.str());
		}
		if (next_nodes.size() == 1) {
			return next_nodes[0];
		}
		for (const auto &e : next_nodes) {
			if (std::get<0>(e) == tt) {
				return e;
			}
		}
		throw std::runtime_error("ERROR : CANNOT INFER NODE");
	}

	/// some code to visualize a graph in graphviz format
	void print_graphviz() {
		std::cout << "digraph G {" << std::endl;
		for (const auto &kv : nodes) {
			for (const auto &e : kv.second.edges) {
				std::cout << kv.first << " -> " << e << std::endl;
			}
		}
		std::cout << "}" << std::endl;
	}

	// handle graph traversal
	void traverse(const T &node) {
		if (traversal_started) {
			if (!check_path(current_node, node, 0, 1)) {
				std::ostringstream buffer;
				buffer << "ERROR : INVALID PATH IN DIGRAPH (" << current_node << " --> " << node << ")";
				throw std::runtime_error(buffer.str());
			}
		}
		current_node = node;
		traversal_started = true;
	}
	void skip(const T &node) {
		if (!traversal_started) {
			throw std::runtime_error("ERROR : CANNOT SKIP BEFORE TRAVERSAL STARTED");
		}
		if (!check_path(current_node, node, 0, 20)) {
			std::ostringstream buffer;
			buffer << "ERROR : INVALID PATH IN DIGRAPH (" << current_node << " --> ... -->" << node << ")";
			throw std::runtime_error(buffer.str());
		}
		current_node = node;
	}

	std::map<T,DIGRAPH_NODE<T>> nodes;
	T current_node;
	bool traversal_started;
};

// how to handle graph traversal

using NODE_TYPE = std::tuple<midi::TOKEN_TYPE,int>;

std::vector<std::vector<midi::TOKEN_TYPE>> DEF_GRAPH = {
  {midi::TOKEN_PIECE_START, midi::TOKEN_NUM_BARS, midi::TOKEN_TRACK},
  {midi::TOKEN_TIME_SIGNATURE, midi::TOKEN_TIME_ABSOLUTE_POS},
  {midi::TOKEN_TIME_SIGNATURE, midi::TOKEN_VELOCITY_LEVEL},
  {midi::TOKEN_TIME_SIGNATURE, midi::TOKEN_FILL_IN_PLACEHOLDER},
  {midi::TOKEN_FILL_IN_PLACEHOLDER, midi::TOKEN_BAR_END},
  {midi::TOKEN_VELOCITY_LEVEL, midi::TOKEN_NOTE_ONSET},
  {midi::TOKEN_VELOCITY_LEVEL, midi::TOKEN_DELTA},
  {midi::TOKEN_DELTA_DIRECTION, midi::TOKEN_DELTA},
  {midi::TOKEN_DELTA, midi::TOKEN_DELTA},
  {midi::TOKEN_DELTA, midi::TOKEN_DELTA_DIRECTION},
  {midi::TOKEN_DELTA, midi::TOKEN_NOTE_ONSET},
  {midi::TOKEN_DELTA, midi::TOKEN_FILL_IN_END},
  {midi::TOKEN_NOTE_ONSET, midi::TOKEN_NOTE_DURATION},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_TIME_ABSOLUTE_POS},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_NOTE_ONSET},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_VELOCITY_LEVEL},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_BAR_END},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_FILL_IN_END},
  {midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_NOTE_ONSET},
  {midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_VELOCITY_LEVEL},
  {midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_BAR_END},
  {midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_FILL_IN_END},
  {midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_DELTA},
  {midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_DELTA_DIRECTION},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_DELTA},
  {midi::TOKEN_NOTE_DURATION, midi::TOKEN_DELTA_DIRECTION},
  {midi::TOKEN_DELTA_DIRECTION, midi::TOKEN_DELTA},
  {midi::TOKEN_BAR_END, midi::TOKEN_BAR},
  {midi::TOKEN_BAR_END, midi::TOKEN_TRACK_END},
  {midi::TOKEN_TRACK_END, midi::TOKEN_TRACK},
  {midi::TOKEN_TRACK_END, midi::TOKEN_FILL_IN_START},
  {midi::TOKEN_FILL_IN_START, midi::TOKEN_TIME_ABSOLUTE_POS},
  {midi::TOKEN_FILL_IN_START, midi::TOKEN_VELOCITY_LEVEL},
  {midi::TOKEN_FILL_IN_END, midi::TOKEN_FILL_IN_START},
};

template<typename T, typename... Us>
std::vector<std::variant<T,Us...>> convert(const std::vector<T> &xs) {
	std::vector<std::variant<T,Us...>> ys;
	for (const auto &x : xs) {
		ys.push_back(x);
	}
	return ys;
}

template<typename T, typename... Us>
std::vector<std::tuple<T,Us...>> convert(std::vector<T> &xs, std::tuple<Us...> defaults) {
	std::vector<std::tuple<T,Us...>> ys;
	for (auto x : xs) {
		ys.push_back(std::tuple_cat(std::make_tuple(x), defaults));
	}
	return ys;
}

template<size_t N, typename T, typename... Us>
std::vector<T> convert(std::vector<std::tuple<Us...>> &xs) {
	std::vector<T> ys;
	for (auto x : xs) {
		ys.push_back(std::get<N>(x));
	}
	return ys;
}

class REP_GRAPH {
public:	
	REP_GRAPH(encoder::ENCODER *e, enums::MODEL_TYPE mt) {
		enc = e;
		initialize(mt, {});
	}
	REP_GRAPH(encoder::ENCODER *e, enums::MODEL_TYPE mt, std::vector<std::tuple<midi::TOKEN_TYPE,int>> tokens_to_remove) {
		enc = e;
		initialize(mt, tokens_to_remove);
	}
	void initialize(enums::MODEL_TYPE mt, std::vector<std::tuple<midi::TOKEN_TYPE,int>> tokens_to_remove) {
		for (auto &path : DEF_GRAPH) {
			graph.add_path(convert(path, std::tuple<int>(0)));
		}
		graph.add_path(encoder::get_bar_attribute_control_graph_v2());
		graph.add_path(encoder::get_track_attribute_control_graph_v2());
		graph.add_path(encoder::get_track_pre_instrument_attribute_control_graph_v2());
		std::vector<NODE_TYPE> to_remove; 
		auto just_tokens = convert<0,midi::TOKEN_TYPE>(tokens_to_remove);
		for (const auto &kv : graph.nodes) {
			if (!enc->rep->has_token_type(std::get<0>(kv.first))) {
				to_remove.push_back(kv.first);
			}
			if (std::find(just_tokens.begin(), just_tokens.end(), std::get<0>(kv.first)) != just_tokens.end()) {
				to_remove.push_back(kv.first);
			}
		}
		graph.remove_nodes(to_remove);

		if (mt == enums::TRACK_MODEL) {
			initialize_autoregressive();
		}
		else {
			initialize_bar_infilling();
		}
	}
	void initialize_bar_infilling() {
		std::set<midi::TOKEN_TYPE> to_keep = {
			midi::TOKEN_VELOCITY_LEVEL,
			midi::TOKEN_NOTE_ONSET,
			midi::TOKEN_NOTE_DURATION,
			midi::TOKEN_DELTA,
			midi::TOKEN_DELTA_DIRECTION,
			midi::TOKEN_TIME_ABSOLUTE_POS,
			midi::TOKEN_FILL_IN_START,
			midi::TOKEN_FILL_IN_END,
		};
		std::vector<NODE_TYPE> to_remove;
		for (const auto &kv : graph.nodes) {
			if (to_keep.find(std::get<0>(kv.first)) == to_keep.end()) {
				to_remove.push_back(kv.first);
			}
		}
		graph.remove_nodes_wo_connecting(to_remove);
		graph.current_node = std::make_tuple(midi::TOKEN_FILL_IN_END,0); // necessary for graph traversal
	}

	void initialize_autoregressive() {
		std::vector<midi::TOKEN_TYPE> to_remove = {
			midi::TOKEN_FILL_IN_PLACEHOLDER,
			midi::TOKEN_FILL_IN_START,
			midi::TOKEN_FILL_IN_END
		};
		graph.remove_nodes_wo_connecting(convert(to_remove, std::tuple<int>(0)));
	}
	void set_mask(int last_token, std::vector<int> &mask) {
		auto tt = graph.infer_node(last_token, enc);
		data_structures::LOGGER(data_structures::to_str("GRAPH INFERENCE : ", toString(tt)));
		graph.traverse(tt); // validate graph traversals
		for (const auto &e : graph.get_next_nodes(tt)) {
			enc->rep->set_mask(std::get<0>(e), {-1}, mask, 1);
		}
	}

	// helpers
	std::vector<midi::TOKEN_TYPE> get_next_nodes(midi::TOKEN_TYPE ttt) {
		NODE_TYPE tt = std::make_tuple(ttt, 0);
		std::vector<NODE_TYPE> next_nodes = graph.get_next_nodes(tt);
		std::vector<midi::TOKEN_TYPE> next_tokens;
		for (const auto &e : next_nodes) {
			next_tokens.push_back(std::get<0>(e));
		}
		return next_tokens;
	}

	std::vector<midi::TOKEN_TYPE> get_previous_nodes(midi::TOKEN_TYPE ttt) {
		NODE_TYPE tt = std::make_tuple(ttt, 0);
		std::vector<NODE_TYPE> next_nodes = graph.get_previous_nodes(tt);
		std::vector<midi::TOKEN_TYPE> next_tokens;
		for (const auto &e : next_nodes) {
			next_tokens.push_back(std::get<0>(e));
		}
		return next_tokens;
	}

	void skip(const midi::TOKEN_TYPE &t) {
		graph.skip(std::make_tuple(t, 0));
	}
 
	encoder::ENCODER *enc;
	DIGRAPH<NODE_TYPE> graph;
};

}