class Admin::BooksController < ApplicationController

    def index
        @books = Book.all

        flash[:notice] = "NOW: #{Time.now}"
    end

    def show
        # NOP
    end

end
