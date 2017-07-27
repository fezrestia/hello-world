require 'test_helper'

class SampleControllerTest < ActionDispatch::IntegrationTest
  # test "the truth" do
  #   assert true
  # end



  # Sample HTML test
  test "test html" do

    html = ""
    html << "<h1>Title</h1>\n"
    html << "<table>\n"
    html << "  <tr>\n"
    html << "    <th>Head1</th>\n"
    html << "    <th>Head2</th>\n"
    html << "    <th>Head3</th>\n"
    html << "  </tr>\n"
    html << "  <tr>\n"
    html << "    <td>Data1</td>\n"
    html << "    <td>Data2</td>\n"
    html << "    <td></td>\n"
    html << "  </tr>\n"
    html << "  <tr>\n"
    html << "    <td>Data11</td>\n"
    html << "    <td></td>\n"
    html << "    <td>Data13</td>\n"
    html << "  </tr>\n"
    html << "</table>\n"


    require 'nokogiri'

    @selected = Nokogiri::HTML(html)

    assert_select "h1", "Title"
    assert_select "table", true
    assert_select "table tr th"
    assert_select "table tr td"

    # header.
    assert_select "table>tr>th" do |headers|
      assert_equal "Head1", headers[0].content
      assert_equal "Head2", headers[1].content
      assert_equal "Head3", headers[2].content
    end

    # Data records.
    assert_select "table>tr>td" do |records|
      assert_equal "Data1", records[0].content
      assert_equal "Data2", records[1].content
      assert_equal "", records[2].content
      assert_equal "Data11", records[3].content
      assert_equal "", records[4].content
      assert_equal "Data13", records[5].content
    end

  end

end
