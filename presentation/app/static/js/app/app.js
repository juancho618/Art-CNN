let app = angular.module('reportApp', ['ngMaterial'])

.config(function($interpolateProvider) {
    $interpolateProvider.startSymbol('//').endSymbol('//');
});