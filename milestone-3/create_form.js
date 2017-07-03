

function getSpreadsheetData(sheetName) {
  // This function gives you an array of objects modeling a worksheet's tabular data, where the first items — column headers — become the property names.
  var arrayOfArrays = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheetName || 'Sheet3').getDataRange().getValues();
  var headers = arrayOfArrays.shift();
  return arrayOfArrays.map(function (row) {
    return row.reduce(function (memo, value, index) {
      if (value) {
        memo[headers[index]] = value;
      }
      return memo;
    }, {});
  });
}

function makeOurForm() {
  var form = FormApp.create('group5 sentiment and emotion analysis')
  
  form.setDescription('This is an example of setting a description for our programmatically generated Form.');
  
  var values = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Sheet3').getDataRange().getValues();
  
  for (n=0;n<values.length;++n) {
    var cell = values[n][0] ;

    form.addSectionHeaderItem()
      .setTitle(cell);

    var item = form.addMultipleChoiceItem();
    item.setTitle('Sentiment')
      .setChoices([
        item.createChoice('highly positive'),
        item.createChoice('moderately positive'),
        item.createChoice('neutral'),
        item.createChoice('moderately negative'),
        item.createChoice('highly negative')]);

    var item = form.addMultipleChoiceItem();
    item.setTitle('Emotion')
      .setChoices([
        item.createChoice('joy'),
        item.createChoice('fear'),
        item.createChoice('anger'),
        item.createChoice('disgust'),
        item.createChoice('sadness'),
        item.createChoice('others')]);

  };
}
