import struct

def encode(postings):
    """_summary_

    Args:
        postings (_type_): _description_

    Returns:
        _type_: _description_
    """
    b = bytearray()
   
    for p in postings:
        # Basic info
        b.extend(struct.pack("IIB", p.doc_id, p.tf, p.is_important))
        
        # Position info (limit to 50 positions to save space)
        positions = p.positions if p.positions else []
        positions = positions[:50]
        
        # Number of positions
        b.extend(struct.pack("H", len(positions)))
        
        # Each position (max value 65535)
        for pos in positions:
            b.extend(struct.pack("H", min(pos, 65535)))
    
    return bytes(b)