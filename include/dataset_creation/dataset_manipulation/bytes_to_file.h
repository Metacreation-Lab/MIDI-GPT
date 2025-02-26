#pragma once

#include <string>
#include <fstream>
#include "../../../libraries/protobuf/build/midi.pb.h"
#include "../compression/lz4.h"


namespace dataset_manipulation {
	class BytesToFile{
	private:
		std::string filepath_;
		std::string header_filepath_;
		std::fstream file_stream_;
		std::fstream header_file_stream_;
		midi::Dataset dataset_split_protobuf_;
		int flush_count_;
		bool can_write;

	public:
		BytesToFile(std::string external_filepath_);
		void enableWrite();
		void appendBytesToFileStream(std::string& bytes_as_string, size_t split_id);
		void writeFile();
		void close();
	};
}