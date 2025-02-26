#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include "../../../libraries/protobuf/build/midi.pb.h"
#include "../../../include/dataset_creation/dataset_manipulation/bytes_to_file.h"
#include "../../../include/dataset_creation/compression/lz4.h"
#include "../../../include/dataset_creation/dataset_manipulation/bytes_to_file.h"

namespace dataset_manipulation {

	BytesToFile::BytesToFile(std::string user_filepath_) {
		filepath_ = user_filepath_;
		header_filepath_ = user_filepath_ + ".header";
		flush_count_ = 0;
		can_write = false;
	}

	void BytesToFile::enableWrite() {
		if (can_write) { return; }
		// check that the current file is empty unless force flag is present ?
		file_stream_.open(filepath_, std::ios::out | std::ios::binary);
		can_write = true;
	}

	void BytesToFile::appendBytesToFileStream(std::string& bytes_as_string, size_t split_id) {
		//file_stream_.open(filepath_, std::ios::out | std::ios::binary);
		enableWrite();
		
		//Start compression ==============================
		size_t stream_position_start = file_stream_.tellp();
		size_t source_size = sizeof(char) * bytes_as_string.size();
		size_t destination_capacity = LZ4_compressBound(source_size);
		char* destination = new char[destination_capacity];
		size_t destination_size = LZ4_compress_default(
			(char*)bytes_as_string.c_str(), destination, source_size, destination_capacity);
		file_stream_.write(destination, destination_size);
		delete[] destination;
		size_t stream_position_end = file_stream_.tellp();
		// end compression ===============================

		midi::Item* item;
		switch (split_id) {
		case 0: item = dataset_split_protobuf_.add_train(); break;
		case 1: item = dataset_split_protobuf_.add_valid(); break;
		case 2: item = dataset_split_protobuf_.add_test(); break;
		}
		item->set_start(stream_position_start);
		item->set_end(stream_position_end);
		item->set_src_size(source_size);
		flush_count_++;

		if (flush_count_ >= 1000) {
			writeFile();
			flush_count_ = 0;
		};
	}

	void BytesToFile::writeFile() {
		file_stream_.flush();
		//TODO: Check if the header stuff actually makes sense... we might not be using the header ever.
		header_file_stream_.open(header_filepath_, std::ios::out | std::ios::binary);
		if (!dataset_split_protobuf_.SerializeToOstream(&header_file_stream_)) {
			std::cerr << "ERROR : Failed to write header file" << std::endl;
		}
		header_file_stream_.close();
	}

	void BytesToFile::close()
	{
		writeFile();
		file_stream_.close();
		header_file_stream_.close();
	}
}

