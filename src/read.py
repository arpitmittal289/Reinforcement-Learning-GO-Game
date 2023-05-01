def readInput(n, path="D:/input_go.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()
        piece_type = int(lines[0])
        
        previous_board = []
        for i in range(5):
            ithRow = []
            for j in range(5):
                ithRow.append(int(lines[i+1][j]))
            previous_board.append(ithRow)
            
        board = []
        for i in range(5):
            ithRow = []
            for j in range(5):
                ithRow.append(int(lines[i+6][j]))
            board.append(ithRow)
            
        return piece_type, previous_board, board

def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y
