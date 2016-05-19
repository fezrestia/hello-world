require 'test_helper'

class ApplicationHelperTest < ActionView::TestCase

  test "full page title helper test" do
    assert_equal 'Title | Ruby on Rails Tutorial Sample App', full_page_title('Title')
    assert_equal 'Ruby on Rails Tutorial Sample App', full_page_title
  end

end

