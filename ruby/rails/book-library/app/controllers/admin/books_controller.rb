class Admin::BooksController < ApplicationController

    def index
        @books = Book.all
    end

    def show
        # NOP
    end

end
