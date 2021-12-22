import numpy as np

WHITE = 1
BLACK = 2
EMPTY = 0


class GameBoard:
    def __init__(self):
        super(GameBoard, self).__init__()
        self.board = np.zeros((8, 8))
        self.board[3][3] = WHITE
        self.board[4][4] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK

    def reset(self):
        self.board = np.zeros((8, 8))
        self.board[3][3] = WHITE
        self.board[4][4] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK

    @staticmethod
    def matrix_to_coordinates(matrix_coord):
        """
        The game board uses coordinates from 1 to 8. The matrix uses
        coordinates from 0 to 7, and with different axes.
        :param matrix_coord: The matrix coordinates, or matrix indices, e.g.
                             matrix_cord = (mx, my)
         with self.board[mx][my].
        :return: Board coordinates x, y
        """
        mx, my = matrix_coord
        x = my + 1
        y = 8 - mx

        return x, y

    @staticmethod
    def coordinates_to_matrix(coord):
        """
        The game board uses coordinates from 1 to 8. The matrix uses
        coordinates from 0 to 7, and with different axes.
        :param coord: The board coordinates
        :return: Matrix coordinates, or matrix indices
        """
        x, y = coord
        mx = 8 - y
        my = x - 1

        return mx, my

    def place_piece(self, coord, player):
        """
        Places a piece if valid. Update board. Check wins?
        :param coord:
        :param player:
        :return: True if piece is placed, False if not
        """
        if self.is_valid_placement(coord, player, suppress_error=False):
            captured_pieces = self.enclosed_pieces(coord, player)
            self.board[self.coordinates_to_matrix(coord)] = player
            for piece in captured_pieces:
                self.board[piece[0], piece[1]] = player
            return True
        return False

    def find_moves(self, player, game_board=None):
        """
        Finds possible moves in matrix coordinates.
        :param player:
        :param game_board:
        :return:
        """
        moves = list()
        if game_board is not None:
            self.board = game_board
        for i in range(8):
            for j in range(8):
                if self.is_valid_placement(
                    self.matrix_to_coordinates((i, j)), player
                ):
                    moves = moves + [(i, j)]
        return moves

    def has_moves(self, player):
        """
        Check if a player has any legal moves left.
        :param player: The player whose moves should be checked
        :return: True or false
        """
        for i in range(8):
            for j in range(8):
                if self.is_valid_placement((i + 1, j + 1), player):
                    return True
        return False

    def is_valid_placement(self, coord, player, suppress_error=True):
        """
        Check if length of enclosed pieces > 0 and coord is empty (==0)
        :param coord: Board coordinates
        :param player:
        :param suppress_error: Whether or not to suppress error messages
        :return: True or false
        """
        x, y = coord
        if x < 1 or y < 1 or x > 8 or y > 8:
            if not suppress_error:
                print("Invalid coordinates")
            return False
        if player != WHITE and player != BLACK:
            if not suppress_error:
                print("Invalid player color")
            return False
        if self.board[self.coordinates_to_matrix(coord)] != EMPTY:
            if not suppress_error:
                print("Board location is not empty")
            return False
        captured_pieces = self.enclosed_pieces(coord, player)
        if len(captured_pieces) > 0:
            return True
        else:
            if not suppress_error:
                print("Invalid move -- does not capture any pieces")

    def closest_n_piece(self, coord, color):
        """
        For a given board coordinate, find the closest northwards piece of the
        desired color.
        :param coord: The board coordinate
        :param color: The color of the piece
        :return: The number of squares until the piece of desired color. A
                 piece directly to the north of the given coordinate has a
                 distance 1. A distance of 0 means there are no correct
                 pieces to the north.
        """
        x, y = coord
        if x < 1 or y < 1 or x > 8 or y > 7:
            return 0
        else:
            for i in range(8 - y):
                if (
                    self.board[self.coordinates_to_matrix((x, y + i + 1))]
                    == color
                ):
                    return i + 1
        return 0

    def closest_e_piece(self, coord, color):
        """
        For a given board coordinate, find the closest eastwards piece of the
        desired color.
        :param coord: The board coordinate
        :param color: The color of the piece
        :return: The number of squares until the piece of desired color. A
                 piece directly to the east of the given coordinate has a
                 distance 1. A distance of 0 means there are no correct
                 pieces to the east.
        """
        x, y = coord
        if x < 1 or y < 1 or x > 7 or y > 8:
            return 0
        else:
            for i in range(8 - x):
                if (
                    self.board[self.coordinates_to_matrix((x + i + 1, y))]
                    == color
                ):
                    return i + 1
        return 0

    def closest_s_piece(self, coord, color):
        """
        For a given board coordinate, find the closest southwards piece of the
        desired color.
        :param coord: The board coordinate
        :param color: The color of the piece
        :return: The number of squares until the piece of desired color. A
                 piece directly to the south of the given coordinate has a
                 distance 1. A distance of 0 means there are no correct
                 pieces to the south.
        """
        x, y = coord
        if x < 1 or y < 2 or x > 8 or y > 8:
            return 0
        else:
            for i in range(y - 1):
                if (
                    self.board[self.coordinates_to_matrix((x, y - i - 1))]
                    == color
                ):
                    return i + 1
        return 0

    def closest_w_piece(self, coord, color):
        """
        For a given board coordinate, find the closest westwards piece of the
        desired color.
        :param coord: The board coordinate
        :param color: The color of the piece
        :return: The number of squares until the piece of desired color. A
                 piece directly to the west of the given coordinate has a
                 distance 1. A distance of 0 means there are no correct
                 pieces to the west.
        """
        x, y = coord
        if x < 2 or y < 1 or x > 8 or y > 8:
            return 0
        else:
            for i in range(x - 1):
                if (
                    self.board[self.coordinates_to_matrix((x - i - 1, y))]
                    == color
                ):
                    return i + 1
        return 0

    def closest_ne_piece(self, coord, color):
        """ """
        x, y = coord
        if x < 1 or y < 1 or x > 7 or y > 7:
            return 0
        else:
            for i in range(min(8 - x, 8 - y)):
                if (
                    self.board[
                        self.coordinates_to_matrix((x + i + 1, y + i + 1))
                    ]
                    == color
                ):
                    return i + 1
        return 0

    def closest_se_piece(self, coord, color):
        """ """
        x, y = coord
        if x < 1 or y < 2 or x > 7 or y > 8:
            return 0
        else:
            for i in range(min(8 - x, y - 1)):
                if (
                    self.board[
                        self.coordinates_to_matrix((x + i + 1, y - i - 1))
                    ]
                    == color
                ):
                    return i + 1
        return 0

    def closest_sw_piece(self, coord, color):
        """ """
        x, y = coord
        if x < 2 or y < 2 or x > 8 or y > 8:
            return 0
        else:
            for i in range(min(x - 1, y - 1)):
                if (
                    self.board[
                        self.coordinates_to_matrix((x - i - 1, y - i - 1))
                    ]
                    == color
                ):
                    return i + 1
        return 0

    def closest_nw_piece(self, coord, color):
        """ """
        x, y = coord
        if x < 2 or y < 1 or x > 8 or y > 7:
            return 0
        else:
            for i in range(min(x - 1, 8 - y)):
                if (
                    self.board[
                        self.coordinates_to_matrix((x - i - 1, y + i + 1))
                    ]
                    == color
                ):
                    return i + 1
        return 0

    def enclosed_pieces(self, coord, player):
        """
        Finds the matrix coordinates of all enclosed pieces
        :param coord: Board coordinates of the placed piece
        :param player: The player placing the piece
        :return: An array of coordinates, in matrix coordinates
        """
        coordinate_list = list()
        x, y = coord
        if x < 1 or y < 1 or x > 8 or y > 8:
            return []
        if player == WHITE:
            opponent = BLACK
        elif player == BLACK:
            opponent = WHITE
        else:
            print("Invalid player color in function enclosed_pieces")
            return []

        # Check north
        north_list = list()
        is_valid = True
        n = self.closest_n_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[self.coordinates_to_matrix((x, y + i + 1))]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[self.coordinates_to_matrix((x, y + i + 1))]
                    == opponent
                ):
                    north_list = north_list + [
                        list(self.coordinates_to_matrix((x, y + i + 1)))
                    ]
        if is_valid:
            coordinate_list = coordinate_list + north_list

        # Check east
        east_list = list()
        is_valid = True
        n = self.closest_e_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[self.coordinates_to_matrix((x + i + 1, y))]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[self.coordinates_to_matrix((x + i + 1, y))]
                    == opponent
                ):
                    east_list = east_list + [
                        list(self.coordinates_to_matrix((x + i + 1, y)))
                    ]
        if is_valid:
            coordinate_list = coordinate_list + east_list

        # Check south
        south_list = list()
        is_valid = True
        n = self.closest_s_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[self.coordinates_to_matrix((x, y - i - 1))]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[self.coordinates_to_matrix((x, y - i - 1))]
                    == opponent
                ):
                    south_list = south_list + [
                        list(self.coordinates_to_matrix((x, y - i - 1)))
                    ]
        if is_valid:
            coordinate_list = coordinate_list + south_list

        # Check west
        west_list = list()
        is_valid = True
        n = self.closest_w_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[self.coordinates_to_matrix((x - i - 1, y))]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[self.coordinates_to_matrix((x - i - 1, y))]
                    == opponent
                ):
                    west_list = west_list + [
                        list(self.coordinates_to_matrix((x - i - 1, y)))
                    ]
        if is_valid:
            coordinate_list = coordinate_list + west_list

        # Check northeast
        northeast_list = list()
        is_valid = True
        n = self.closest_ne_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[
                        self.coordinates_to_matrix((x + i + 1, y + i + 1))
                    ]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[
                        self.coordinates_to_matrix((x + i + 1, y + i + 1))
                    ]
                    == opponent
                ):
                    northeast_list = northeast_list + [
                        list(
                            self.coordinates_to_matrix((x + i + 1, y + i + 1))
                        )
                    ]
        if is_valid:
            coordinate_list = coordinate_list + northeast_list

        # Check southeast
        southeast_list = list()
        is_valid = True
        n = self.closest_se_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[
                        self.coordinates_to_matrix((x + i + 1, y - i - 1))
                    ]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[
                        self.coordinates_to_matrix((x + i + 1, y - i - 1))
                    ]
                    == opponent
                ):
                    southeast_list = southeast_list + [
                        list(
                            self.coordinates_to_matrix((x + i + 1, y - i - 1))
                        )
                    ]
        if is_valid:
            coordinate_list = coordinate_list + southeast_list

        # Check southwest
        southwest_list = list()
        is_valid = True
        n = self.closest_sw_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[
                        self.coordinates_to_matrix((x - i - 1, y - i - 1))
                    ]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[
                        self.coordinates_to_matrix((x - i - 1, y - i - 1))
                    ]
                    == opponent
                ):
                    southwest_list = southwest_list + [
                        list(
                            self.coordinates_to_matrix((x - i - 1, y - i - 1))
                        )
                    ]
        if is_valid:
            coordinate_list = coordinate_list + southwest_list

        # Check northwest
        northwest_list = list()
        is_valid = True
        n = self.closest_nw_piece(coord, player)
        if n > 1:
            for i in range(n - 1):
                if (
                    self.board[
                        self.coordinates_to_matrix((x - i - 1, y + i + 1))
                    ]
                    == EMPTY
                ):
                    is_valid = False
                    break
                elif (
                    self.board[
                        self.coordinates_to_matrix((x - i - 1, y + i + 1))
                    ]
                    == opponent
                ):
                    northwest_list = northwest_list + [
                        list(
                            self.coordinates_to_matrix((x - i - 1, y + i + 1))
                        )
                    ]
        if is_valid:
            coordinate_list = coordinate_list + northwest_list

        return coordinate_list

    def get_board(self):
        """
        Copies the board and returns it
        :return: The copied board
        """

        return np.copy(self.board)
