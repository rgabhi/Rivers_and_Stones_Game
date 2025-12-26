import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time



class Piece:
    def __init__(self, owner:str, side:str="stone", orientation:Optional[str]=None):
        self.owner = owner
        self.side = side
        self.orientation = orientation if orientation else "horizontal"
    def copy(self): return Piece(self.owner, self.side, self.orientation)
    def to_dict(self): return {"owner":self.owner,"side":self.side,"orientation":self.orientation}
    @staticmethod
    def from_dict(d:Optional[Dict[str,Any]]):
        if d is None: return None
        return Piece(d["owner"], d.get("side","stone"), d.get("orientation","horizontal"))

# ==================== GAME UTILITIES ====================


def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows


def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""

    if cols <= 12:
        w = 4
    elif cols <= 14:
        w = 5
    else:
        w = 6
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))


def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2


def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"

# ==================== MOVE GENERATION HELPERS ====================

def get_valid_moves_for_piece(board, x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all valid moves for a specific piece.

    Args:
        board: Current board state
        x, y: Piece position
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices

    Returns:
        List of valid move dictionaries
    """
    moves = []
    piece = board[y][x]

    if piece is None or piece.owner != player:
        return moves

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    if piece.side == "stone":
        # Stone movement
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue

            if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                continue

            if board[ny][nx] is None:
                # Simple move
                moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
            elif board[ny][nx].owner != player:
                # Push move
                px, py = nx + dx, ny + dy
                if (in_bounds(px, py, rows, cols) and
                        board[py][px] is None and
                        not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})

        # Stone to river flips
        for orientation in ["horizontal", "vertical"]:
            moves.append({"action": "flip", "from": [x, y], "orientation": orientation})

    else:  # River piece
        # River to stone flip
        moves.append({"action": "flip", "from": [x, y]})

        # River rotation
        moves.append({"action": "rotate", "from": [x, y]})

    return moves


# ==================== RIVER FLOW SIMULATION ====================

def agent_river_flow(board, rx: int, ry: int, sx: int, sy: int, player: str,
                     rows: int, cols: int, score_cols: List[int], river_push: bool = False) -> List[Tuple[int, int]]:
    """
    Simulate river flow from a given position.

    Args:
        board: Current board state
        rx, ry: River entry point
        sx, sy: Source position (where piece is moving from)
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices
        river_push: Whether this is for a river push move

    Returns:
        List of (x, y) coordinates where the piece can end up via river flow
    """
    destinations = []
    visited = set()
    queue = [(rx, ry)]

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[y][x]
        if river_push and x == rx and y == ry:

            if in_bounds(sx, sy, rows, cols):
                cell = board[sy][sx]
            else:
                continue

        if cell is None:
            if is_opponent_score_cell(x, y, player, rows, cols, score_cols):

                pass
            else:
                destinations.append((x, y))
            continue

        if getattr(cell, "side", "stone") != "river":
            continue


        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue

                if getattr(next_cell, "side", "stone") == "river":
                    queue.append((nx, ny))
                    break
                break


    unique_destinations = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            unique_destinations.append(d)

    return unique_destinations


# ==================== MOVE VALIDATION AND GENERATION ====================

def agent_compute_valid_moves(board, sx: int, sy: int, player: str, rows: int, cols: int, score_cols: List[int]) -> Dict[str, Any]:
    """
    Compute all valid moves for a piece at position (sx, sy).

    Returns:
        Dictionary with 'moves' (set of coordinates) and 'pushes' (list of tuples)
    """
    if not in_bounds(sx, sy, rows, cols):
        return {'moves': set(), 'pushes': []}

    piece = board[sy][sx]
    if piece is None or piece.owner != player:
        return {'moves': set(), 'pushes': []}

    moves = set()
    pushes = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in directions:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue


        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue

        target = board[ty][tx]

        if target is None:

            moves.add((tx, ty))
        elif getattr(target, "side", "stone") == "river":

            flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for dest in flow:
                moves.add(dest)
        else:

            pushed_player = target.owner
            if getattr(piece, "side", "stone") == "stone":

                px, py = tx + dx, ty + dy
                if (in_bounds(px, py, rows, cols) and
                        board[py][px] is None and
                        not is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols)): # GoodAI_Code: Check against pushed_player's score cell
                    pushes.append(((tx, ty), (px, py)))
            else:

                flow = agent_river_flow(board, tx, ty, sx, sy, pushed_player, rows, cols, score_cols, river_push=True) # GoodAI_Code: Pass pushed_player
                for dest in flow:
                    if not is_opponent_score_cell(dest[0], dest[1], pushed_player, rows, cols, score_cols): # GoodAI_Code: Check against pushed_player's score cell
                        pushes.append(((tx, ty), dest))

    return {'moves': moves, 'pushes': pushes}


# ==================== MOVE APPLICATION (FOR SIMULATION) ====================

def agent_apply_move(board, move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, str]:
    """
    Apply a move to a board copy for simulation purposes.

    Args:
        board: Board state to modify
        move: Move dictionary
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices

    Returns:
        (success: bool, message: str)
    """
    action = move.get("action")

    if action == "move":
        return _apply_move_action(board, move, player, rows, cols, score_cols)
    elif action == "push":
        return _apply_push_action(board, move, player, rows, cols, score_cols)
    elif action == "flip":
        return _apply_flip_action(board, move, player, rows, cols, score_cols)
    elif action == "rotate":
        return _apply_rotate_action(board, move, player, rows, cols, score_cols)

    return False, "unknown action"


def _apply_move_action(board, move, player, rows, cols, score_cols):
    """Apply a move action."""
    fr = move.get("from")
    to = move.get("to")
    if not fr or not to:
        return False, "bad move format"

    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])

    if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
        return False, "out of bounds"

    if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
        return False, "cannot move into opponent score cell"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if board[ty][tx] is None:
        # Simple move
        board[ty][tx] = piece
        board[fy][fx] = None
        return True, "moved"


    pushed_to = move.get("pushed_to")
    if not pushed_to:

        return False, "destination occupied"


    ptx, pty = int(pushed_to[0]), int(pushed_to[1])

    target_piece = board[ty][tx]
    if target_piece is None:
        return False, "move with push 'to' not occupied"

    pushed_player = target_piece.owner

    dx, dy = tx - fx, ty - fy
    if (ptx, pty) != (tx + dx, ty + dy):

        if piece.side == "river":
            return False, "invalid pushed_to for river move"

        if (ptx, pty) != (tx + dx, ty + dy):
            return False, "invalid pushed_to for stone push"


    if not in_bounds(ptx, pty, rows, cols):
        return False, "pushed_to out of bounds"


    if is_opponent_score_cell(ptx, pty, pushed_player, rows, cols, score_cols):
        return False, "cannot push into opponent score"

    if board[pty][ptx] is not None:
        return False, "pushed_to not empty"

    board[pty][ptx] = board[ty][tx]
    board[ty][tx] = piece
    board[fy][fx] = None
    return True, "moved with push"


def _apply_push_action(board, move, player, rows, cols, score_cols):
    """Apply a push action."""
    fr = move.get("from")
    to = move.get("to")
    pushed_to = move.get("pushed_to")

    if not fr or not to or not pushed_to:
        return False, "bad push format"

    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])
    px, py = int(pushed_to[0]), int(pushed_to[1])

    if not (in_bounds(fx, fy, rows, cols) and
            in_bounds(tx, ty, rows, cols) and
            in_bounds(px, py, rows, cols)):
        return False, "out of bounds"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"

    target_piece = board[ty][tx]
    if target_piece is None:
        return False, "'to' must be occupied"

    pushed_player = target_piece.owner

    if (is_opponent_score_cell(tx, ty, player, rows, cols, score_cols) or
            is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols)):
        return False, "push would move into opponent score cell"

    if board[py][px] is not None:
        return False, "pushed_to not empty"


    if piece.side == "river" and target_piece.side == "river":
        return False, "rivers cannot push rivers"

    board[py][px] = board[ty][tx]
    board[ty][tx] = board[fy][fx]
    board[fy][fx] = None


    mover = board[ty][tx]
    if mover.side == "river":
        mover.side = "stone"
        mover.orientation = None

    return True, "pushed"


def _apply_flip_action(board, move, player, rows, cols, score_cols):
    """Apply a flip action."""
    fr = move.get("from")
    if not fr:
        return False, "bad flip format"

    fx, fy = int(fr[0]), int(fr[1])
    if not in_bounds(fx, fy, rows, cols):
        return False, "out of bounds"

    piece = board[fy][fx]

    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if piece.side == "stone":
        # Stone to river
        orientation = move.get("orientation")
        if orientation not in ("horizontal", "vertical"):
            return False, "stone->river needs orientation"

        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)


        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None

        for dx, dy in flow:
            if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
                return False, "flip would allow flow into opponent score cell"


        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        return True, "flipped to river"
    else:

        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None
        return True, "flipped to stone"


def _apply_rotate_action(board, move, player, rows, cols, score_cols):
    """Apply a rotate action."""
    fr = move.get("from")
    if not fr:
        return False, "bad rotate format"

    fx, fy = int(fr[0]), int(fr[1])
    if not in_bounds(fx, fy, rows, cols):
        return False, "out of bounds"

    piece = board[fy][fx]

    if piece is None or piece.owner != player or piece.side != "river":
        return False, "invalid rotate"


    old_orientation = piece.orientation
    piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"


    flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)

    for dx, dy in flow:
        if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):

            piece.orientation = old_orientation
            return False, "rotate would allow flow into opponent score cell"

    return True, "rotated"


def generate_board_hash(board: List[List[Optional[Piece]]]) -> Tuple:
    """Create a hashable representation of the board state."""
    return tuple(
        tuple((p.owner, p.side, p.orientation) if p else None for p in row)
        for row in board
    )


# ==================== BOARD EVALUATION ====================

def count_stones_in_scoring_area(board: List[List[Optional[Piece]]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """counting stones a player has in its scoring area."""

    count = 0

    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)

    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1

    return count



#I have implemented this , It counts number of pieces in scoring area ,same as count_stones_in_scoring_area function.
def count_rivers_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int,
                                 score_cols: List[int]) -> int:
    """Count how many stones a player has in their scoring area."""
    count = 0

    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)

    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "river":
                count += 1

    return count


# I have implemented this -> It maps player to own-row,opponent-row and cols ->not using right now
def map_score(player: str, rows: int,cols: int) -> Dict[str, Dict[str, Any]]:

    opponent = get_opponent(player)
    scoring_cols = score_cols_for(cols)

    mapping = {
        player: {
            "own_row": top_score_row() if player == "circle" else bottom_score_row(rows),
            "opp_row": bottom_score_row(rows) if player == "circle" else top_score_row(),
            "cols": scoring_cols
        },
        opponent: {
            "own_row": top_score_row() if opponent == "circle" else bottom_score_row(rows),
            "opp_row": bottom_score_row(rows) if opponent == "circle" else top_score_row(),
            "cols": scoring_cols
        }
    }

    return mapping


# I have implemented this -> It gives distance of player or opponent to score area ->manhattan distance
def get_distance(player: str, rows:int, cols: int, position_player_rows:int, position_player_cols:int) -> int:

    target_col = score_cols_for(cols)

    if player == "circle":
        target_row = top_score_row()
    else:
        target_row = bottom_score_row(rows)

    total_col = float("inf")
    for x in target_col:
        total_col = min(total_col,abs(position_player_cols - x))
    distance = (abs(target_row - position_player_rows) + total_col)

    return distance


# This function will give number of stone and river for player
def count_rivers_and_stone(board: List[List[Any]], player: str, rows: int, cols: int)-> (float,float) :

    stones = 0
    rivers = 0

    for x in range(rows):
        for y in range(cols):
            piece = board[x][y]
            if piece is not None:
                if piece.owner == player:
                    if piece.side == "river":
                        rivers += 1
                    else:
                        stones += 1

    return stones, rivers


# I have implemented this -> It maps from player to its respective rows and columns and also same for opponent
# Currently not using it
def player_mapping(board:List[List[Any]] ,player:str,rows:int,cols:int) -> Dict[str, List[Tuple[int, int]]]:
    player_map: Dict[str, List[Tuple[int, int]]] = {player:[],get_opponent(player):[]}
    opponent= get_opponent(player)

    for x in range(rows):
        for y in range(cols):
            piece = board[x][y]
            if piece is not None and hasattr(piece, 'owner'):
                if piece.owner == player :
                    player_map[player].append((x,y))
                elif piece.owner == opponent:
                    player_map[opponent].append((x,y))

    return player_map


def eval_board(board: List[List[Optional[Piece]]], rows: int, cols: int, score_cols: List[int],
               a: int, b: int, original: str) -> float:
    """
    I have implemented this , It is improved version of basic_evaluate_board function
    """
    opponent = get_opponent(original)

    player_map_data = player_mapping(board, original, rows, cols)
    player_map = player_map_data.get(original, [])
    opponent_map = player_map_data.get(opponent, [])

    distance = float("inf")
    sum_dist = 0.0

    player_piece_count = len(player_map)
    if player_piece_count > 0:
        for play in player_map:
            dist = get_distance(original,rows,cols,play[0],play[1])
            if dist != 0:
             distance = min(distance,dist)
            sum_dist += dist
        average = sum_dist/player_piece_count
    else:
        distance = 99
        average = 99

    # Heuristic to make sure pieces move towards scoring area
    h1 = a * distance + b * average

    distance_opponent = float("inf")
    sum_opponent = 0.0

    opponent_piece_count = len(opponent_map)

    # Heuristic to make sure opponent pieces are not moving towards player scoring area
    if opponent_piece_count > 0:
        for play in opponent_map:
            dist = get_distance(opponent, rows, cols, play[0], play[1])
            if dist != 0:
             distance_opponent = min(distance_opponent, dist)
            sum_opponent += dist
        average_opponent = sum_opponent / opponent_piece_count
    else:
        distance_opponent = 99
        average_opponent = 99

    h2 = a * distance_opponent + b * average_opponent

    total_stones_player_scoring = count_stones_in_scoring_area(board,original,rows,cols,score_cols)
    total_stones_opponent_scoring=count_stones_in_scoring_area(board,opponent,rows,cols,score_cols)

    # Heuristic to give preference to stones in scoring area
    h3 = 1000 * total_stones_player_scoring
    h3 = h3 - 1000 * total_stones_opponent_scoring

    total_rivers_player_scoring = count_rivers_in_scoring_area(board,original,rows,cols,score_cols)
    total_rivers_opponent_scoring = count_rivers_in_scoring_area(board,opponent,rows,cols,score_cols)

    h4 = 60 * (total_rivers_player_scoring - total_rivers_opponent_scoring)

    h_stones, h_river = count_rivers_and_stone(board, original, rows,cols)

    h5 = 100 if abs(h_stones-h_river) <= 6 else -10

    scoring_river_bonus = 0

    if total_rivers_player_scoring > 0:
        scoring_river_bonus = total_rivers_player_scoring * 1000

    total_score = (h2 - h1) + h3 + h4 + h5 + scoring_river_bonus

    return total_score


# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """

    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)

    @abstractmethod
    def choose(self, board: List[List[Optional[Piece]]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.

        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas

        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass


def apply_brute_force(board:List[List[Optional[Piece]]], player, rows: int, cols:int, score_cols: List[int]) -> Dict[str,Any]:
    """
    This is function to return move of making river to stone in scoring area .
    """
    x = top_score_row() if player == "circle" else bottom_score_row(rows)
    move = None
    for y in score_cols:
        piece = board[x][y]
        if piece and piece.owner == player:
            if piece.side == "river":
                move = {"action": "flip", "from": [y, x]}
    return move


# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):

    def __init__(self, player: str):
        super().__init__(player)
        self.brute_force = False
        self.turn_ = 1
        self.count_ = 0
        self.dp: Dict[Tuple, Dict] = {}
        self.start_time = 0.0
        self.time_limit = 0.0

        self.MAX_DEPTH = 50

        self.TIME_MARGIN = 0.2

    def generate_all_moves(self, board: List[List[Optional[Piece]]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """
        proposes all possible moves; validity (specially for flips/rotates) will be checked by simulate_move.
        """
        moves = []
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece or piece.owner != player:
                    continue


                valid_targets = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)

                for (tx, ty) in valid_targets.get('moves', set()):
                    moves.append({"action": "move", "from": [x, y], "to": [tx, ty]})

                for (to_pos, pushed_to_pos) in valid_targets.get('pushes', []):
                    moves.append({
                        "action": "push",
                        "from": [x, y],
                        "to": [to_pos[0], to_pos[1]],
                        "pushed_to": [pushed_to_pos[0], pushed_to_pos[1]]
                    })


                if piece.side == "stone":

                    moves.append({"action": "flip", "from": [x, y], "orientation": "horizontal"})
                    moves.append({"action": "flip", "from": [x, y], "orientation": "vertical"})

                else:
                    moves.append({"action": "flip", "from": [x, y]})

                    moves.append({"action": "rotate", "from": [x, y]})

        return moves

    def simulate_move(self, board: List[List[Optional[Piece]]], move: Dict[str, Any], player: str, rows: int, cols: int,
                      score_cols: List[int]) -> Tuple[bool, Any]:
        """
        simulate a move on a copy of the board.

        Returns:
            (success: bool, new_board or error_message)
        """
        board_copy = copy.deepcopy(board)

        success, message = agent_apply_move(board_copy, move, player, rows, cols, score_cols)

        if success:
            return True, board_copy
        else:
            return False, message

    def recursive_mini_max(self,
                           brute_force: bool,
                           depth: int,
                           board: List[List[Optional[Piece]]],
                           original_player: str,
                           current_player: str,
                           rows: int, cols: int, score_cols: List[int],
                           alpha: float, beta: float,
                           is_maximizing: bool,
                           a: int, b: int) -> float:
        """
         min-imax
         added alpha-beta Pruning and
         dp.
        """

        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Search time exceeded")
        if depth == 0:
            return eval_board(board, rows, cols, score_cols, a, b, original_player)

        board_hash = generate_board_hash(board)
        if board_hash in self.dp:
            stored_entry = self.dp[board_hash]

            if stored_entry['depth'] >= depth:
                return stored_entry['score']

        opponent_player = get_opponent(current_player)

        if is_maximizing:
            best_score = float("-inf")
            possible_moves = self.generate_all_moves(board, current_player, rows, cols, score_cols)


            possible_moves.sort(key=lambda m: (
                    m.get('action') == 'flip' and
                    m.get('from') and
                    in_bounds(m['from'][0], m['from'][1], rows, cols) and
                    board[m['from'][1]][m['from'][0]] is not None and
                    board[m['from'][1]][m['from'][0]].side == 'river' and
                    is_own_score_cell(m['from'][0], m['from'][1], current_player, rows, cols, score_cols)
            ), reverse=True)

            if not possible_moves:
                return eval_board(board, rows, cols, score_cols, a, b, original_player)

            for move in possible_moves:
                success, new_board = self.simulate_move(board, move, current_player, rows, cols, score_cols)
                if success:
                    value = self.recursive_mini_max(brute_force, depth - 1, new_board, original_player, opponent_player,
                                                    rows, cols, score_cols, alpha, beta, False, a, b)
                    best_score = max(best_score, value)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break

            self.dp[board_hash] = {'depth': depth, 'score': best_score}
            return best_score

        else:
            best_score = float("inf")
            possible_moves = self.generate_all_moves(board, current_player, rows, cols, score_cols)

            possible_moves.sort(key=lambda m: (
                    m.get('action') == 'flip' and
                    m.get('from') and
                    in_bounds(m['from'][0], m['from'][1], rows, cols) and
                    board[m['from'][1]][m['from'][0]] is not None and
                    board[m['from'][1]][m['from'][0]].side == 'river' and
                    is_own_score_cell(m['from'][0], m['from'][1], current_player, rows, cols, score_cols)
            ), reverse=True)

            if not possible_moves:
                return eval_board(board, rows, cols, score_cols, a, b, original_player)

            for move in possible_moves:
                success, new_board = self.simulate_move(board, move, current_player, rows, cols, score_cols)
                if success:
                    value = self.recursive_mini_max(brute_force, depth - 1, new_board, original_player, opponent_player,
                                                    rows, cols, score_cols, alpha, beta, True, a, b)
                    best_score = min(best_score, value)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

            self.dp[board_hash] = {'depth': depth, 'score': best_score}
            return best_score

    def init_flip(self, board, player, cols):
        moves = None
        if cols == 12 and self.count_ in (0, 1, 2, 3):
            x = 4 if player == "square" else 8
            y = 5 + self.count_
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        elif cols == 14 and self.count_ in (0, 1, 2, 3,4):
            x = 4 if player == "square" else 10
            y = 5 + self.count_
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        elif cols == 16 and self.count_ in (0, 1, 2, 3,4,5):
            x = 4 if player == "square" else 12
            y = 6 + self.count_
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        return moves

    def initial_board_changes_small(self, board: List[List[Optional[Piece]]], player) -> Dict[str, Any]:
        moves = None
        turn = self.turn_
        # 8,5

        if turn in (1, 2, 3):
            x = 4 if player == "square" else 8
            y = 4 - turn
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "move", "from": [y, x], "to": [y - 1, x]}

        if turn == 4:
            x = 4 if player == "square" else 8
            y = 0
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "vertical"}

        if turn == 5:
            x = 4 if player == "square" else 8
            y = 4
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":

                    if p.side == "stone":
                        moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        if turn in (6, 7, 8):
            x = 3 if player == "square" else 9
            y = 9 - turn
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "move", "from": [y, x], "to": [y - 1, x]}

        if turn == 9:
            x = 3 if player == "square" else 9
            y = 0
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        if turn == 10:
            x = 3 if player == "square" else 9
            y = 0
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [0, 12]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [0, 0]}

        if turn == 11:
            x = 4 if player == "square" else 8
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [7, 12]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [7, 0]}

        if turn == 12:
            x = 4 if player == "square" else 8
            y = 4
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 13:
            x = 4 if player == "square" else 8
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [6, 12]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [6, 0]}

        if turn == 14:
            x = 4 if player == "square" else 8
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 15:
            x = 4 if player == "square" else 8
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [5, 12]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [5, 0]}

        if turn == 16:
            x = 4 if player == "square" else 8
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 17:
            x = 4 if player == "square" else 8
            y = 8
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [4, 12]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [4, 0]}

        if turn in (18, 19):
            x = 12 - (turn - 18) if player == "square" else (turn - 18)
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (20, 21):
            x = 12 - (turn - 20) if player == "square" else turn - 20
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (22, 23):
            x = 12 - (turn - 22) if player == "square" else turn - 22
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (24, 25):
            x = 12 - (turn - 24) if player == "square" else turn - 24
            y = 4
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        return moves

    def initial_board_changes_medium(self, board: List[List[Optional[Piece]]], player) -> Dict[str, Any]:
        moves = None
        turn = self.turn_
        if turn in (1, 2, 3):
            x = 4 if player == "square" else 10
            y = 4 - turn
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "move", "from": [y, x], "to": [y - 1, x]}

        if turn == 4:
            x = 4 if player == "square" else 10
            y = 0
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "vertical"}

        if turn == 5:
            x = 4 if player == "square" else 10
            y = 4
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":

                    if p.side == "stone":
                        moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        if turn in (6, 7, 8):
            x = 3 if player == "square" else 11
            y = 9 - turn
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "move", "from": [y, x], "to": [y - 1, x]}

        if turn == 9:
            x = 3 if player == "square" else 11
            y = 0
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        if turn == 10:
            x = 3 if player == "square" else 11
            y = 0
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [0, 14]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [0, 0]}

        if turn == 11:
            x = 4 if player == "square" else 10
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                # if p.side in ("stone", "river"):
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [8, 14]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [8, 0]}

        if turn == 12:
            x = 4 if player == "square" else 10
            y = 4
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 13:
            x = 4 if player == "square" else 10
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [7, 14]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [7, 0]}

        if turn == 14:
            x = 4 if player == "square" else 10
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 15:
            x = 4 if player == "square" else 10
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [6, 14]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [6, 0]}

        if turn == 16:
            x = 4 if player == "square" else 10
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 17:
            x = 4 if player == "square" else 10
            y = 8
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [5, 14]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [5, 0]}

        if turn == 18:
            x = 4 if player == "square" else 10
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 19:
            x = 4 if player == "square" else 10
            y = 9
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [4, 14]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [4, 0]}

        if turn in (20, 21):
            x = 14 - (turn - 20) if player == "square" else (turn - 20)
            y = 8
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (22, 23):
            x = 14 - (turn - 22) if player == "square" else turn - 22
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (24, 25):
            x = 14 - (turn - 24) if player == "square" else turn - 24
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (26, 27):
            x = 14 - (turn - 26) if player == "square" else turn - 26
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (28, 29):
            x = 14 - (turn - 28) if player == "square" else turn - 28
            y = 4
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        return moves

    def initial_board_changes_large(self, board: List[List[Optional[Piece]]], player) -> Dict[str, Any]:
        moves = None
        turn = self.turn_
        if turn in (1, 2, 3, 4):
            x = 4 if player == "square" else 12
            y = 5 - turn
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "move", "from": [y, x], "to": [y - 1, x]}

        if turn == 5:
            x = 4 if player == "square" else 12
            y = 0
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "vertical"}

        if turn == 6:
            x = 4 if player == "square" else 12
            y = 5
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":

                    if p.side == "stone":
                        moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        if turn in (7, 8, 9, 10):
            x = 3 if player == "square" else 13
            y = 11 - turn
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "move", "from": [y, x], "to": [y - 1, x]}

        if turn == 11:
            x = 3 if player == "square" else 13
            y = 0
            p = board[x][y]

            if p and p.owner == player:
                if p.side == "stone":
                    moves = {"action": "flip", "from": [y, x], "orientation": "horizontal"}

        if turn == 12:
            x = 3 if player == "square" else 13
            y = 0
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [0, 16]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [0, 0]}

        if turn == 13:
            x = 4 if player == "square" else 12
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [10, 16]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [10, 0]}

        if turn == 14:
            x = 4 if player == "square" else 12
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 15:
            x = 4 if player == "square" else 12
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [9, 16]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [9, 0]}

        if turn == 16:
            x = 4 if player == "square" else 12
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 17:
            x = 4 if player == "square" else 12
            y = 8
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [8, 16]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [8, 0]}

        if turn == 18:
            x = 4 if player == "square" else 12
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 19:
            x = 4 if player == "square" else 12
            y = 9
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [7, 16]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [7, 0]}

        if turn == 20:
            x = 4 if player == "square" else 12
            y = 8
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 21:
            x = 4 if player == "square" else 12
            y = 10
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [6, 16]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [6, 0]}
        if turn == 22:
            x = 4 if player == "square" else 12
            y = 9
            p = board[x][y]
            if p and p.owner == player:
                if p.side == "river":
                    # Flip stone → river
                    if player == "square":
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}
                    else:
                        moves = {"action": "move", "from": [y, x], "to": [y + 1, x]}

        if turn == 23:
            x = 4 if player == "square" else 12
            y = 11
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [5, 16]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [5, 0]}

        if turn in (24, 25):
            x = 16 - (turn - 24) if player == "square" else (turn - 24)
            y = 10
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (26, 27):
            x = 16 - (turn - 26) if player == "square" else turn - 26
            y = 9
            p = board[x][y]
            if p and p.owner == player:
                # if p.side == "stone":
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (28, 29):
            x = 16 - (turn - 28) if player == "square" else turn - 28
            y = 8
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (30, 31):
            x = 16 - (turn - 30) if player == "square" else turn - 30
            y = 7
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (32, 33):
            x = 16 - (turn - 32) if player == "square" else turn - 32
            y = 6
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        if turn in (34, 35):
            x = 16 - (turn - 34) if player == "square" else turn - 34
            y = 5
            p = board[x][y]
            if p and p.owner == player:
                if player == "square":
                    moves = {"action": "move", "from": [y, x], "to": [y, x - 1]}
                else:
                    moves = {"action": "move", "from": [y, x], "to": [y, x + 1]}

        return moves

    def choose(self, board: List[List[Optional[Piece]]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        choose the best move for the curr board.
        """
        best_move = None
        self.dp.clear()
        self.start_time = time.time()


        time_alloc_factor = 40.0
        floor_time = 0.5
        ceiling_time = 5.0

        allocated_time = current_player_time / time_alloc_factor
        max_time_for_move = max(floor_time, min(allocated_time, ceiling_time))

        self.time_limit = max_time_for_move - self.TIME_MARGIN

        if self.count_ <= 5:
            best_move = self.init_flip(board,self.player,cols)
            self.count_ += 1
        # If not possible to move then make best_move to none
        if best_move is not None:
            success, new_board = self.simulate_move(board, best_move, self.player, rows, cols, score_cols)
            if success:
                return best_move
            else:
                self.count_ = 10

        # Here I have hardcoded to make sure I win in at most 35 moves
        if self.turn_ <= 35:
            if cols == 12:
                best_move = self.initial_board_changes_small(board, self.player)
            elif cols == 14:
                best_move = self.initial_board_changes_medium(board, self.player)
            elif cols == 16:
                best_move = self.initial_board_changes_large(board, self.player)

            # If not possible to move then make best_move to none
            if best_move is not None:
                self.turn_ += 1
                success, new_board = self.simulate_move(board, best_move, self.player, rows, cols, score_cols)
                if success:
                    return best_move
                else:
                    self.turn_ = 60
            else:
                self.turn_ = 60

        stones_in_win = count_stones_in_scoring_area(board, self.player, rows, cols, score_cols)
        rivers_in_win = count_rivers_in_scoring_area(board, self.player, rows, cols, score_cols)

        win_count = 4
        if cols <= 12:
            win_count = 4
        elif cols <= 14:
            win_count = 5
        else:
            win_count = 6

        # If all river are present at scoring area then make it to stone,it is brute force
        if (stones_in_win + rivers_in_win == win_count) and rivers_in_win > 0:
            best_move = apply_brute_force(board, self.player, rows, cols, score_cols)
            if best_move:
                return best_move

        best_move_found_at_any_depth = None

        try:

            possible_moves = self.generate_all_moves(board, self.player, rows, cols, score_cols)

            if not possible_moves:
                return None

            best_move_found_at_any_depth = random.choice(possible_moves)

            for depth in range(1, self.MAX_DEPTH):
                best_score_for_this_depth = float("-inf")
                current_best_move = None

                if best_move:
                    possible_moves.sort(key=lambda m: m == best_move, reverse=True)

                for move in possible_moves:
                    success, new_board = self.simulate_move(board, move, self.player, rows, cols, score_cols)

                    if success:

                        score = self.recursive_mini_max(
                            self.brute_force,
                            depth - 1,
                            new_board,
                            self.player,
                            self.opponent,
                            rows, cols, score_cols,
                            float("-inf"),
                            float("inf"),
                            False,
                            20, 5
                        )

                        if score > best_score_for_this_depth:
                            best_score_for_this_depth = score
                            current_best_move = move

                if current_best_move:
                    best_move_found_at_any_depth = current_best_move

        except TimeoutError:
            pass

        except Exception as e:
            pass

        return best_move_found_at_any_depth


# ==================== TESTING HELPERS ====================

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")

    try:
        # Try to import from gameEngine if available
        try:
            from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
            rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        except ImportError:
            # Fallback if gameEngine.py is not in path
            rows, cols = 13, 12

        score_cols = score_cols_for(cols)

        # Create a simple board manually if import fails
        board = [[None for _ in range(cols)] for __ in range(rows)]
        board[rows-4][cols//2] = Piece("circle", "stone") # A single piece

        agent = StudentAgent("circle")
        # Give it a reasonable amount of time for a test
        move = agent.choose(board, rows, cols, score_cols, 60.0, 60.0)

        if move:
            print(f"✓ Agent successfully generated a move: {move}")
        else:
            # It's possible no valid moves exist from the simple board,
            # but it should at least not crash.
            print("✓ Agent returned no move (or completed search)")

    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully (gameEngine not found, basic create test only)")
    except Exception as e:
        print(f"✗ Agent test failed with error: {e}")


if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()