import unittest
import os
from game.state import GameState
from ai.opening_book import OpeningBook

class TestMCTSAndOpeningBook(unittest.TestCase):
    def test_opening_book_loading_and_lookup(self):
        book = OpeningBook()
        self.assertGreater(len(book.book), 0)
        
        # Test lookup with default setup state
        state = GameState()
        state.default_setup()
        
        # Check if default setup has an opening book move
        book_move = book.get_move(state)
        # Note: Depending on whether default state is in book, get_move returns int or None
        if book_move is not None:
            self.assertIsInstance(book_move, int)
            self.assertGreaterEqual(book_move, 0)
            self.assertLess(book_move, 959)

if __name__ == '__main__':
    unittest.main()
