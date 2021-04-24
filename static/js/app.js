var data = [];
var token = "";

jQuery(document).ready(function () {
  $("#input_question").keyup(function (e) {
    if (e.which === 13) {
      $("#btn-process").click();
    }
  });

  $("#btn-process").on("click", function () {
    input_question = $("#input_idx").val();

    $.ajax({
      url: "/get_answer",
      type: "post",
      contentType: "application/json",
      dataType: "json",
      data: JSON.stringify({
        input_idx: input_question,
      }),
      beforeSend: function () {
        $(".overlay").show();
        $("#question").val("");
        $("#text_paragraphs").val("");
        $("#bm25").val("");
        $("#tfidf").val("");
        $("#doc2vec").val("");
        $("#sbert").val("");
        $("#stbart").val("");
        $("#mtbart").val("");
      },
      complete: function () {
        $(".overlay").hide();
      },
    })
      .done(function (jsondata, textStatus, jqXHR) {
        console.log(jsondata);
        $("#question").val(jsondata["question"]);
        $("#text_paragraphs").val(jsondata["reviews"]);
        $("#bm25").val(jsondata["bm25"]);
        $("#tfidf").val(jsondata["tf-idf"]);
        $("#doc2vec").val(jsondata["doc2vec"]);
        $("#sbert").val(jsondata["s-bert"]);
        $("#stbart").val(jsondata["st-bart"]);
        $("#mtbart").val(jsondata["mt-bart"]);
      })
      .fail(function (jsondata, textStatus, jqXHR) {
        console.log(jsondata);
        alert(jsondata["responseText"]);
      });
  });
});
