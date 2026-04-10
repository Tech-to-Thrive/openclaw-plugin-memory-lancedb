const heading = "Light Sleep";

// Match heading + all lines (including empty) that DON'T start with "## "
const sectionRegex = new RegExp(`^## ${heading}\\n(?:(?!## ).*\\n?)*`, "m");
console.log("Pattern:", sectionRegex.source);

const content = "## Light Sleep\nSome text\nMore text\n\n## REM Sleep\nOther text";
const match = content.match(sectionRegex);
console.log("Test 1 - matched:", JSON.stringify(match?.[0]));

const content2 = "## Light Sleep\nSome text\nMore text\n";
const match2 = content2.match(sectionRegex);
console.log("Test 2 - no following:", JSON.stringify(match2?.[0]));

const content3 = "## Deep Sleep\nOld\n\n## Light Sleep\nSome text\n\n## REM Sleep\nOther";
const match3 = content3.match(sectionRegex);
console.log("Test 3 - middle section:", JSON.stringify(match3?.[0]));

const content4 = "## Light Sleep\nSome text\n\nBlank line above\n\n## REM Sleep\nOther";
const match4 = content4.match(sectionRegex);
console.log("Test 4 - with blanks:", JSON.stringify(match4?.[0]));

// Replacement tests
const nextBlock = "## Light Sleep\nNew content\n\n";
const updated = content.replace(sectionRegex, nextBlock);
console.log("Test 5 - replaced:", JSON.stringify(updated));

const updated3 = content3.replace(sectionRegex, nextBlock);
console.log("Test 6 - replaced middle:", JSON.stringify(updated3));

const updated4 = content4.replace(sectionRegex, nextBlock);
console.log("Test 7 - replaced with blanks:", JSON.stringify(updated4));
