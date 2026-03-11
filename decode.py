import struct
def decode(data):
    """
    Decoding the raw binary to text

    Args:
        data (Posting): Posting (doc id, term frequency, important)
    
    Return - list(Posting):
        a list of postings 
    """
    postings = []
    offset = 0
    
    while offset < len(data):
        # Read basic info (9 bytes: 4 + 4 + 1)
        if offset + 9 > len(data):
            break
        
        doc_id, tf, important = struct.unpack("IIB", data[offset:offset+9])
        offset += 9
        
        # Read number of positions (2 bytes)
        if offset + 2 > len(data):
            postings.append((doc_id, tf, important, []))
            break
        
        num_positions = struct.unpack("H", data[offset:offset+2])[0]
        offset += 2
        
        # Read positions
        positions = []
        for _ in range(num_positions):
            if offset + 2 > len(data):
                break
            pos = struct.unpack("H", data[offset:offset+2])[0]
            positions.append(pos)
            offset += 2
        
        postings.append((doc_id, tf, important, positions))
    
    return postings