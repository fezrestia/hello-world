class User < ActiveRecord::Base
    has_one :book
end

